#include "EvFDaqDirector.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FastMonitoringService.h"
#include <iostream>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>

using std::stringstream;

namespace evf {

namespace {
  struct flock make_flock(short type, short whence, off_t start, off_t len, pid_t pid)
  {
#ifdef __APPLE__
    return {start, len, pid, type, whence};
#else
    return {type, whence, start, len, pid};
#endif
  }
}

EvFDaqDirector::EvFDaqDirector(const edm::ParameterSet &pset,
		edm::ActivityRegistry& reg) :
			testModeNoBuilderUnit_(
					pset.getUntrackedParameter<bool> ("testModeNoBuilderUnit",
							false)),
			base_dir_(
					pset.getUntrackedParameter<std::string> ("baseDir",
							"/tmp/hlt")),
			run_dir_("0"),
			bu_base_dir_(
					pset.getUntrackedParameter<std::string> ("buBaseDir", "bu")),
			sm_base_dir_(
					pset.getUntrackedParameter<std::string> ("smBaseDir", "sm")),
			monitor_base_dir_(
					pset.getUntrackedParameter<std::string> ("monBaseDir",
							"MON")),
			directorBu_(
					pset.getUntrackedParameter<bool> ("directorIsBu", false)),
			copyRunDirToFUs_(
					pset.getUntrackedParameter<bool> ("copyRunDir", false)),
			bu_w_flk( make_flock( F_WRLCK, SEEK_SET, 0, 0, 0 )),
			bu_r_flk( make_flock( F_RDLCK, SEEK_SET, 0, 0, 0 )),
			bu_w_fulk( make_flock( F_UNLCK, SEEK_SET, 0, 0, 0 )),
			bu_r_fulk( make_flock( F_UNLCK, SEEK_SET, 0, 0, 0 ))
			//, fu_rw_flk({F_RDLCK,SEEK_SET,0,0,0})
			,
			fu_rw_flk( make_flock ( F_WRLCK, SEEK_SET, 0, 0, getpid() )),
			fu_rw_fulk( make_flock( F_UNLCK, SEEK_SET, 0, 0, getpid() )),
			bu_w_lock_stream(0),
			bu_r_lock_stream(0),
			fu_rw_lock_stream(0),
			dirManager_(base_dir_),
			slaveResources_(
					pset.getUntrackedParameter<std::vector<std::string>> (
							"slaveResources", std::vector<std::string>())),
			slavePathToData_(
					pset.getUntrackedParameter<std::string> ("slavePathToData",
							"/data")), bu_w_monitor_stream(0),
			bu_t_monitor_stream(0), previousFileSize_(0), jumpLS_(0),
			jumpIndex_(0) {
	reg.watchPreBeginRun(this, &EvFDaqDirector::preBeginRun);
	reg.watchPostEndRun(this, &EvFDaqDirector::postEndRun);

	// check if base dir exists or create it accordingly
	int retval =
			mkdir(base_dir_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
		throw cms::Exception("DaqDirector") << " Error creating base dir "
				<< base_dir_ << " mkdir error:" << strerror(errno) << "\n";
	}

}

void EvFDaqDirector::postEndRun(edm::Run const& run, edm::EventSetup const& es) {
	close(bu_readlock_fd_);
	close(bu_writelock_fd_);
	if (directorBu_) {
		std::string filename = bu_base_dir_ + "/bu.lock";
		removeFile(filename);
	}
}

void EvFDaqDirector::preBeginRun(edm::RunID const& id, edm::Timestamp const& ts) {
	std::cout << "DaqDirector - preBeginRun " << std::endl;
	std::ostringstream ost;

	// save run dir name
	ost << "/run" << id.run();
	run_dir_name_ = ost.str();
	ost.str("");

	// check if run dir exists.
	ost << base_dir_ << "/run" << id.run();
	run_dir_ = ost.str();
	umask(0);
	int retval = mkdir(run_dir_.c_str(),
			S_IRWXU | S_IRWXG | S_IROTH | S_IRWXO | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
		throw cms::Exception("DaqDirector") << " Error creating run dir "
				<< run_dir_ << " mkdir error:" << strerror(errno) << "\n";
	}
	ost.clear();
	ost.str("");
	// check that the requested run is the latest one - this must be imposed at all times
	if (dirManager_.findHighestRunDir() != run_dir_) {
		throw cms::Exception("DaqDirector") << " Error checking run dir "
				<< run_dir_ << " this is not the highest run "
				<< dirManager_.findHighestRunDir() << "\n";
	}

	//create or access  lock files: only the BU director is allowed to create a bu lock, only the FU director is allowed
	// to create a fu lock
	stringstream buRunDirPath;
	buRunDirPath << "/run" << id.run()/* << "/bu"*/;
	bu_run_dir_ = bu_base_dir_ + buRunDirPath.str();
	std::string fulockfile = bu_run_dir_ + "/fu.lock";
	std::string bulockfile = bu_run_dir_ + "/bu.lock";

	if (directorBu_) {
		//make or find bu base dir
		if (bu_base_dir_.empty()) {
			ost << base_dir_ << "/run" << id.run();
			bu_base_dir_ = ost.str();
		} else {
			ost << base_dir_ << "/run" << id.run()/* << "/bu"*/;
			bu_base_dir_ = ost.str();
			retval = mkdir(bu_base_dir_.c_str(),
					S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			if (retval != 0 && errno != EEXIST) {
				throw cms::Exception("DaqDirector")
						<< " Error creating bu dir " << bu_base_dir_
						<< " mkdir error:" << strerror(errno) << "\n";
			}
		}
		ost.clear();
		ost.str("");
		ost << bu_run_dir_ << "/open";
		bu_base_open_dir_ = ost.str();
		retval = mkdir(bu_base_open_dir_.c_str(),
				S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (retval != 0 && errno != EEXIST) {
			throw cms::Exception("DaqDirector") << " Error creating bu dir "
					<< bu_base_open_dir_ << " mkdir error:" << strerror(errno)
					<< "\n";
		}
		ost.clear();
		ost.str("");

		//make or find monitor base dir
		//char thishost[256];
		//gethostname(thishost,255);
		ost << bu_run_dir_ << "/" << monitor_base_dir_;
		monitor_base_dir_ = ost.str() + "_OLD";
		retval = mkdir(monitor_base_dir_.c_str(),
				S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (retval != 0 && errno != EEXIST) {
			throw cms::Exception("DaqDirector")
					<< " Error creating monitor dir " << monitor_base_dir_
					<< " mkdir error:" << strerror(errno) << "\n";
		}

		// the BU director does not need to know about the fu lock
		bu_writelock_fd_ = open(bulockfile.c_str(),
				O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
		if (bu_writelock_fd_ == -1)
			std::cout << "problem with creating filedesc for buwritelock "
					<< strerror(errno) << "\n";
		else
			std::cout << "creating filedesc for buwritelock "
					<< bu_writelock_fd_ << "\n";
		bu_w_lock_stream = fdopen(bu_writelock_fd_, "w");
		if (bu_w_lock_stream == 0)
			std::cout << "Error creating write lock stream " << strerror(errno)
					<< std::endl;
		std::string filename = monitor_base_dir_ + "/bumonitor.txt";
		bu_w_monitor_stream = fopen(filename.c_str(), "w+");
		filename = monitor_base_dir_ + "/diskmonitor.txt";
		bu_t_monitor_stream = fopen(filename.c_str(), "w+");
		if (bu_t_monitor_stream == 0)
			std::cout << "Error creating bu write lock stream " << strerror(
					errno) << std::endl;

		// BU INITIALIZES LOCK FILE
		// FU LOCK FILE OPEN
		openFULockfileStream(fulockfile, true);
		tryInitializeFuLockFile();
		fflush(fu_rw_lock_stream);
		close(fu_readwritelock_fd_);
		if (copyRunDirToFUs_)
			mkFuRunDir();
		createOutputDirectory();
	} else {
		openFULockfileStream(fulockfile, false);
	}

	std::cout << "DaqDirector - preBeginRun success" << std::endl;
}

/*
 bool EvFDaqDirector::copyRunDirToSlaves() {
 if (slaveResources_.size() == 0) {
 return false;
 }

 for (unsigned int i = 0; i < slaveResources_.size(); i++) {
 std::string dataPathInSlave = slaveResources_[i] + ":"
 + slavePathToData_;
 std::string systemCommand = "cp -r " + run_dir_ + " "
 + slavePathToData_;
 //std::string systemCommand = "scp -r " + run_dir_ + " " + dataPathInSlave;
 int rc = system(systemCommand.c_str());
 std::cout << "tried push run dir: " << run_dir_
 << " to slave location: " << dataPathInSlave;
 std::cout << " return code = " << rc << std::endl;
 }
 return true;
 }
 */

bool EvFDaqDirector::mkFuRunDir() {
	std::string runDirInSlave = slavePathToData_ + run_dir_name_;
	int retval = mkdir(runDirInSlave.c_str(),
			S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
		throw cms::Exception("DaqDirector") << " Error creating FU run dir "
				<< runDirInSlave << " mkdir error:" << strerror(errno) << "\n";
		return false;
	}
	return true;
}

bool EvFDaqDirector::createOutputDirectory() {
	int retval = mkdir(sm_base_dir_.c_str(),
			S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
		throw cms::Exception("DaqDirector") << " Error creating output dir "
				<< sm_base_dir_ << " mkdir error:" << strerror(errno) << "\n";
		return false;
	}
	std::string mergedRunDir = sm_base_dir_ + run_dir_name_;
	std::string mergedDataDir = mergedRunDir + "/data";
	std::string mergedMonDir = mergedRunDir + "/mon";
	retval = mkdir(mergedRunDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
		throw cms::Exception("DaqDirector")
				<< " Error creating merged Run dir " << mergedDataDir
				<< " mkdir error:" << strerror(errno) << "\n";
		return false;
	}
	retval
			= mkdir(mergedDataDir.c_str(),
					S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
		throw cms::Exception("DaqDirector")
				<< " Error creating merged data dir " << mergedDataDir
				<< " mkdir error:" << strerror(errno) << "\n";
		return false;
	}
	retval = mkdir(mergedMonDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
		throw cms::Exception("DaqDirector")
				<< " Error creating merged mon dir " << mergedMonDir
				<< " mkdir error:" << strerror(errno) << "\n";
		return false;
	}
	return true;
}

std::string EvFDaqDirector::getFileForLumi(unsigned int ls) {
	std::string retval = bu_base_open_dir_;
	retval += "/ls";
	std::ostringstream ost;
	ost << std::setfill('0') << std::setw(6) << ls << ".raw";
	retval += ost.str();
	return retval;
}

std::string EvFDaqDirector::getWorkdirFileForLumi(unsigned int ls,
		unsigned int index) {
	std::string retval = bu_base_open_dir_;
	retval += "/ls";
	std::ostringstream ost;
	ost << std::setfill('0') << std::setw(4) << ls << "_index" << std::setfill(
			'0') << std::setw(6) << index << ".raw";
	retval += ost.str();
	return retval;
}

std::string EvFDaqDirector::getPathForFU() {
	return sm_base_dir_;
}

void EvFDaqDirector::removeFile(std::string &filename) {
	int retval = remove(filename.c_str());
	if (retval != 0)
		std::cout << "Could not remove used file " << filename << " error "
				<< strerror(errno) << "\n";
	// TODO remove
	printf("OPEN bu_t_monitor_stream\n");
}

void EvFDaqDirector::removeFile(unsigned int ls) {
	std::string filename = getFileForLumi(ls);
	removeFile(filename);
}

void EvFDaqDirector::updateBuLock(unsigned int ls) {
	int check = 0;
	fcntl(bu_writelock_fd_, F_SETLKW, &bu_w_flk);
	if (bu_w_lock_stream != 0) {
		check = fseek(bu_w_lock_stream, 0, SEEK_SET);
		if (check == 0)
			fprintf(bu_w_lock_stream, "%u", ls);
		else
			std::cout
					<< "seek on bu write lock for updating failed with error "
					<< strerror(errno) << std::endl;
	} else
		std::cout << "bu write lock stream is invalid " << strerror(errno)
				<< std::endl;

	fcntl(bu_writelock_fd_, F_SETLKW, &bu_w_fulk);
}

bool EvFDaqDirector::updateFuLock(unsigned int& ls, unsigned int& index,
		bool& eorSeen) {
	// close and reopen the stream / fd
	close(fu_readwritelock_fd_);
	std::string fulockfile = bu_run_dir_ + "/fu.lock";
	fu_readwritelock_fd_ = open(fulockfile.c_str(), O_RDWR, S_IRWXU);
	if (fu_readwritelock_fd_ == -1)
		std::cout << "problem with creating filedesc for fuwritelock "
				<< strerror(errno) << "\n";
	else
		std::cout << "creating filedesc for fureadwritelock "
				<< fu_readwritelock_fd_ << "\n";
	fu_rw_lock_stream = fdopen(fu_readwritelock_fd_, "r+");
	std::cout << "Reopened the fw FD & STREAM" << std::endl;

	// obtain lock on the fulock file
	fcntl(fu_readwritelock_fd_, F_SETLKW, &fu_rw_flk);
	// TODO remove
	timeval theTime;
	gettimeofday(&theTime, 0);
	std::cout << "LOCK aqcuired: s=" << theTime.tv_sec << ": ms = "
			<< theTime.tv_usec / 1000.0 << std::endl;

	eorSeen = false;
	struct stat buf;
	stringstream ss;
	ss << bu_run_dir_ << "/EoR.jsd";
	eorSeen = (stat(ss.str().c_str(), &buf) == 0);
	bool valid = false;

	// if the stream is readable
	if (fu_rw_lock_stream != 0) {
		unsigned int readLs, readIndex, jumpLs, jumpIndex;
		int check = 0;
		// rewind the stream
		check = fseek(fu_rw_lock_stream, 0, SEEK_SET);
		// if rewinded ok
		if (check == 0) {
			// read its' values
			if (testModeNoBuilderUnit_) {
				fscanf(fu_rw_lock_stream, "%u %u %u %u", &readLs, &readIndex,
						&jumpLs, &jumpIndex);
				std::cout << "Read from file: " << readLs << ":" << readIndex
						<< "-->" << jumpLs << ":" << jumpIndex << std::endl;
			} else {
				fscanf(fu_rw_lock_stream, "%u %u", &readLs, &readIndex);
				std::cout << "Read from file: " << readLs << ":" << readIndex
						<< std::endl;
			}

			// try to bump
			bool bumpedOk = bumpFile(readLs, readIndex);
			std::cout << "Bumped to (next file to take): " << readLs << ":"
					<< readIndex << " OK? " << bumpedOk << std::endl;

			ls = readLs;
			// there is a new file to grab or lumisection ended
			if (bumpedOk) {
				index = readIndex;
				// rewind and clear
				check = fseek(fu_rw_lock_stream, 0, SEEK_SET);
				if (check == 0) {
					ftruncate(fu_readwritelock_fd_, 0);
					fflush(fu_rw_lock_stream);
				} else
					std::cout
							<< "seek on fu read/write lock for updating failed with error "
							<< strerror(errno) << std::endl;
				// write new
				// TODO flush already does rewind?!
				check = fseek(fu_rw_lock_stream, 0, SEEK_SET);
				if (check == 0) {
					// write next index in the file, which is the file the next process should take
					if (testModeNoBuilderUnit_) {
						fprintf(fu_rw_lock_stream, "%u %u %u %u", readLs,
								readIndex + 1, readLs + 2, readIndex + 1);
						jumpLS_ = readLs + 2;
						jumpIndex_ = readIndex;
					} else {
						fprintf(fu_rw_lock_stream, "%u %u", readLs,
								readIndex + 1);
					}
					fflush(fu_rw_lock_stream);
					fsync(fu_readwritelock_fd_);

					valid = true;

					if (testModeNoBuilderUnit_)
						std::cout << "Written to file: " << readLs << ":"
								<< readIndex + 1 << " --> " << readLs + 2
								<< ":" << readIndex + 1 << std::endl;
					else
						std::cout << "Written to file: " << readLs << ":"
								<< readIndex + 1 << std::endl;

				} else
					std::cout
							<< "seek on fu read/write lock for updating failed with error "
							<< strerror(errno) << std::endl;
			} else {
				valid = false;
			}

		} else
			std::cout
					<< "seek on fu read/write lock for updating failed with error "
					<< strerror(errno) << std::endl;
	} else
		std::cout << "fu read/write lock stream is invalid " << strerror(errno)
				<< std::endl;

	fcntl(fu_readwritelock_fd_, F_SETLKW, &fu_rw_fulk);

	gettimeofday(&theTime, 0);
	std::cout << "LOCK released: s=" << theTime.tv_sec << ": ms = "
			<< theTime.tv_usec / 1000.0 << std::endl;

	return valid;
}

int EvFDaqDirector::readBuLock() {
	int retval = -1;
	// check if lock file has disappeared and if so whether directory is empty (signal end of run)
	if (!bulock() && dirManager_.checkDirEmpty(bu_base_dir_))
		return retval;
	if (fcntl(bu_readlock_fd_, F_SETLKW, &bu_r_flk) != 0)
		retval = 0;
	if (bu_r_lock_stream) {
		unsigned int readval;
		int check = 0;
		unsigned int *p = &readval;
		check = fseek(bu_r_lock_stream, 0, SEEK_SET);
		if (check == 0) {
			fscanf(bu_r_lock_stream, "%u", p);
			retval = readval;
		}
	} else {
		std::cout << "error reading lock file " << strerror(errno) << std::endl;
		retval = -1;
	}
	fcntl(bu_readlock_fd_, F_SETLKW, &bu_r_fulk);
	std::cout << "readbulock returning " << retval << std::endl;
	return retval;
}

/*
 int EvFDaqDirector::readFuLock() {
 int retval = -1;

 if (!fulock() && dirManager_.checkDirEmpty(bu_base_dir_))
 return retval;
 if (fcntl(fu_readwritelock_fd_, F_SETLKW, &fu_rw_flk) != 0)
 retval = 0;
 if (fu_rw_lock_stream) {
 unsigned int readval;
 int check = 0;
 unsigned int *p = &readval;
 check = fseek(fu_rw_lock_stream, 0, SEEK_SET);
 if (check == 0) {
 fscanf(fu_rw_lock_stream, "%u", p);
 retval = readval;
 }
 } else {
 std::cout << "error reading lock file " << strerror(errno) << std::endl;
 retval = -1;
 }
 fcntl(fu_readwritelock_fd_, F_SETLKW, &fu_rw_fulk);
 std::cout << "readfulock returning " << retval << std::endl;
 return retval;
 }
 */

bool EvFDaqDirector::bulock() {
	struct stat buf;
	std::string lockfile = bu_base_dir_;
	lockfile += "/bu.lock";
	bool retval = (stat(lockfile.c_str(), &buf) == 0);
	if (!retval) {
		close(bu_readlock_fd_);
		close(bu_writelock_fd_);
	}
	std::cout << "stat of lockfile returned " << retval << std::endl;
	return retval;
}

bool EvFDaqDirector::fulock() {
	struct stat buf;
	std::string lockfile = bu_base_dir_;
	lockfile += "/fu.lock";
	bool retval = (stat(lockfile.c_str(), &buf) == 0);
	if (!retval) {
		close(fu_readwritelock_fd_);
		close(fu_readwritelock_fd_);
	}
	std::cout << "stat of lockfile returned " << retval << std::endl;
	return retval;
}

void EvFDaqDirector::writeLsStatisticsBU(unsigned int ls, unsigned int events,
		unsigned long long totsize, long long lsusec) {
	if (bu_w_monitor_stream != 0) {
		int check = fseek(bu_w_monitor_stream, 0, SEEK_SET);
		if (check == 0) {
			fprintf(bu_w_monitor_stream, "%u %u %llu %f %f %012lld", ls,
					events, totsize,
					double(totsize) / double(events) / 1000000.,
					double(totsize) / double(lsusec), lsusec);
			fflush(bu_w_monitor_stream);
		} else
			std::cout
					<< "seek on bu write monitor for updating failed with error "
					<< strerror(errno) << std::endl;
	} else
		std::cout << "bu write monitor stream is invalid " << strerror(errno)
				<< std::endl;

}
void EvFDaqDirector::writeDiskAndThrottleStat(double fraction, int highWater,
		int lowWater) {
	if (bu_t_monitor_stream != 0) {
		int check = fseek(bu_t_monitor_stream, 0, SEEK_SET);
		if (check == 0)
			fprintf(bu_t_monitor_stream, "%f %d %d", fraction, highWater,
					lowWater);
		else
			std::cout
					<< "seek on disk write monitor for updating failed with error "
					<< strerror(errno) << std::endl;
	} else
		std::cout << "disk write monitor stream is invalid " << strerror(errno)
				<< std::endl;

}

bool EvFDaqDirector::bumpFile(unsigned int& ls, unsigned int& index) {
	if (previousFileSize_ != 0) {
		FastMonitoringService *mss = (FastMonitoringService *) (edm::Service<
				evf::MicroStateService>().operator->());
		mss->accummulateFileSize(previousFileSize_);
		previousFileSize_ = 0;
	}
	struct stat buf;
	std::stringstream ss;
	unsigned int nextIndex = index;
	nextIndex++;

	// 1. Check suggested file
	ss << bu_run_dir_ << "/ls" << std::setfill('0') << std::setw(4) << ls
			<< "_index" << std::setfill('0') << std::setw(6) << index << ".raw";
	// std::cout << "Statting --> " << ss.str() << std::endl;
	bool found = (stat(ss.str().c_str(), &buf) == 0);
	// if found
	if (found) {
		//grabbedFileSize = buf.st_size;
		previousFileSize_ = buf.st_size;
		return true;
	}

	// 2. No file -> lumi ended? (and how many?)
	else {
		bool eolFound = true;
		unsigned int currentLumi = ls;
		unsigned int startingLumi = ls;
		while (eolFound) {
			ss.str("");
			ss << bu_run_dir_ << "/EoLS_" << std::setfill('0') << std::setw(4)
					<< currentLumi << ".jsn";
			// std::cout << "Statting --> " << ss.str() << std::endl;
			eolFound = (stat(ss.str().c_str(), &buf) == 0);
			if (eolFound) {
				// this lumi ended, check for files
				++currentLumi;
				ss.str("");
				ss << bu_run_dir_ << "/ls" << std::setfill('0') << std::setw(4)
						<< currentLumi << "_index" << std::setfill('0')
						<< std::setw(6) << 0 << ".raw";
				// std::cout << "Statting --> " << ss.str() << std::endl;
				found = (stat(ss.str().c_str(), &buf) == 0);
				// update highest ls even if there is no file
				// input source can now end its' LS when an EoL jsn file is seen
				ls = currentLumi;
				if (found) {
					// a new file was found at new lumisection, index 0
					index = 0;
					//grabbedFileSize = buf.st_size;
					previousFileSize_ = buf.st_size;

					if (testModeNoBuilderUnit_) {
						// rename ended lumi to + 2
						ss.str("");
						ss << bu_run_dir_ << "/EoLS_" << std::setfill('0')
								<< std::setw(4) << startingLumi << ".jsn";
						string sourceEol = ss.str();

						ss.str("");
						ss << bu_run_dir_ << "/EoLS_" << std::setfill('0')
								<< std::setw(4) << startingLumi + 2 << ".jsn";
						string destEol = ss.str();

						string cpCmd = "cp " + sourceEol + " " + destEol;
						std::cout << " Running copy cmd = " << cpCmd
								<< std::endl;
						int rc = system(cpCmd.c_str());
						if (rc != 0) {
							std::cout << "COPY EOL FAILED!!!!!: " << cpCmd
									<< std::endl;
						}
					}

					return true;
				}
			}
		}
	}
	// no new file found
	return false;
}

std::string EvFDaqDirector::findHighestRunDirStem() {
	boost::filesystem::path highestRunDirPath (findHighestRunDir());
	return highestRunDirPath.filename().string();
}

void EvFDaqDirector::tryInitializeFuLockFile() {
	if (fu_rw_lock_stream == 0)
		std::cout << "Error creating fu read/write lock stream " << strerror(
				errno) << std::endl;
	else {
		std::cout << "Initializing FU LOCK FILE" << std::endl;
		unsigned int readLs = 1, readIndex = 0, jumpLs = 3, jumpIndex = 0;
		if (testModeNoBuilderUnit_)
			fprintf(fu_rw_lock_stream, "%u %u %u %u", readLs, readIndex,
					jumpLs, jumpIndex);
		else
			fprintf(fu_rw_lock_stream, "%u %u", readLs, readIndex);
	}
}

void EvFDaqDirector::openFULockfileStream(std::string& fulockfile, bool create) {
	if (create) {
		fu_readwritelock_fd_ = open(fulockfile.c_str(), O_RDWR | O_CREAT,
				S_IRWXU | S_IRWXG | S_IROTH | S_IRWXO | S_IXOTH);
	} else {
		fu_readwritelock_fd_ = open(fulockfile.c_str(), O_RDWR, S_IRWXU);
	}
	if (fu_readwritelock_fd_ == -1)
		std::cout << "problem with creating filedesc for fuwritelock "
				<< strerror(errno) << "\n";
	else
		std::cout << "creating filedesc for fureadwritelock "
				<< fu_readwritelock_fd_ << "\n";
	fu_rw_lock_stream = fdopen(fu_readwritelock_fd_, "r+");
}

}
