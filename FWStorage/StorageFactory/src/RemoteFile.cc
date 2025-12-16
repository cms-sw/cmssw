#include "FWStorage/StorageFactory/interface/RemoteFile.h"
#include "FWStorage/StorageFactory/src/Throw.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <sys/wait.h>
#include <sys/types.h>
#include <cerrno>
#include <cassert>
#include <spawn.h>
#include <unistd.h>
#include <ostream>
#include <cstring>
#include <filesystem>
#include <vector>
#if __APPLE__
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#endif

using namespace edm::storage;

static std::string join(char **cmd) {
  size_t size = 0;
  for (char **p = cmd; p && p[0]; ++p)
    size += 1 + strlen(*p);

  std::string result;
  result.reserve(size);

  for (char **p = cmd; p && p[0]; ++p) {
    if (p != cmd)
      result += ' ';
    result += *p;
  }

  return result;
}

RemoteFile::RemoteFile(IOFD fd, const std::string &name) : File(fd), name_(name) {}

void RemoteFile::remove(void) { unlink(name_.c_str()); }

void RemoteFile::close(void) {
  remove();
  File::close();
}

void RemoteFile::abort(void) {
  remove();
  File::abort();
}

int RemoteFile::local(const std::string &tmpdir, std::string &temp) {
  // Download temporary files to the current directory by default.
  // This is better for grid jobs as the current directory is
  // likely to have more space, and is more optimised for
  // large files, and is cleaned up after the job.
  std::vector<char> temp_chars;

  if (tmpdir.empty() || tmpdir == ".") {
    std::filesystem::path current_path = std::filesystem::current_path();
    auto spath = current_path.string();
    temp_chars.reserve(spath.size() + 30);
    temp_chars.assign(spath.begin(), spath.end());
  } else {
    temp_chars.reserve(tmpdir.size() + 30);
    temp_chars.assign(tmpdir.begin(), tmpdir.end());
  }

  if (temp_chars.back() != '/')
    temp_chars.push_back('/');

  std::string suffix = "storage-factory-local-XXXXXX";
  temp_chars.insert(temp_chars.end(), suffix.begin(), suffix.end());

  // Ensure the vector is null-terminated
  temp_chars.push_back('\0');

  int fd = mkstemp(temp_chars.data());
  if (fd == -1)
    throwStorageError("RemoteFile", "Calling RemoteFile::local()", "mkstemp()", errno);
  // Copy temp_chars to temp
  temp.assign(temp_chars.begin(), temp_chars.end() - 1);  // Exclude the null terminator

  return fd;
}

std::unique_ptr<Storage> RemoteFile::get(int localfd, const std::string &name, char **cmd, int mode) {
  // FIXME: On write, create a temporary local file open for write;
  // on close, trigger transfer to destination.  If opening existing
  // file for write, may need to first download.
  assert(!(mode & (IOFlags::OpenWrite | IOFlags::OpenCreate)));

  pid_t pid = -1;
  int rc = posix_spawnp(&pid, cmd[0], nullptr, nullptr, cmd, environ);

  if (rc == -1) {
    int errsave = errno;
    ::close(localfd);
    unlink(name.c_str());
    throwStorageError("RemoteFile", "Calling RemoteFile::get()", "posix_spawnp()", errsave);
  }

  pid_t rcpid;
  do
    rcpid = waitpid(pid, &rc, 0);
  while (rcpid == (pid_t)-1 && errno == EINTR);

  if (rcpid == (pid_t)-1) {
    int errsave = errno;
    ::close(localfd);
    unlink(name.c_str());
    throwStorageError("RemoteFile", "Calling RemoteFile::get()", "waitpid()", errsave);
  }

  if (WIFEXITED(rc) && WEXITSTATUS(rc) == 0)
    return std::unique_ptr<Storage>(static_cast<Storage *>(new RemoteFile(localfd, name)));
  else {
    ::close(localfd);
    unlink(name.c_str());
    cms::Exception ex("RemoteFile");
    ex << "'" << join(cmd) << "'"
       << (WIFEXITED(rc)     ? " exited with exit code "
           : WIFSIGNALED(rc) ? " died from signal "
                             : " died for an obscure unknown reason with exit status ")
       << (WIFEXITED(rc)     ? WEXITSTATUS(rc)
           : WIFSIGNALED(rc) ? WTERMSIG(rc)
                             : rc);
    ex.addContext("Calling RemoteFile::get()");
    throw ex;
  }
}
