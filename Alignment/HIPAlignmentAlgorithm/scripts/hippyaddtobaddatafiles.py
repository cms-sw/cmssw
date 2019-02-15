#!/usr/bin/env python

from __future__ import print_function

import argparse, contextlib, os, re, shutil, subprocess, tempfile, time

if __name__ == "__main__":
  def abspath(path):
    if not os.path.exists(path): raise ValueError(path+" does not exist")
    return os.path.abspath(path)

  p = argparse.ArgumentParser()
  p.add_argument("cfgfile", type=abspath)
  p.add_argument("baddatafileslist", nargs="?", default=None)
  args = p.parse_args()

def runcfg(cfgfile, badfilelist):
  try:
    subprocess.check_output(["cmsRun", cfgfile], stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    if "FallbackFileOpenError" in e.output:
      output = e.output.split("An exception of category 'FallbackFileOpenError' occurred while")[1]
      filename = re.search("Failed to open the file '[^']*(/store/.*[.]root)", output).group(1)
      with OneAtATime(badfilelist+".tmp", 2) as f:
        with open(badfilelist) as f:
          contents = set(f.read().split())
        if filename in contents:
          raise RuntimeError(filename+"\nis already in\n"+badfilelist+"\n\nExiting to avoid an infinite loop.  Maybe you have this running on the same cfg file multiple times?")
        contents.add(filename)
        contents = sorted(contents)
        with open(badfilelist, "w") as f:
          f.write("\n".join(contents)+"\n")
        print("found and added a bad file:\n"+filename)
    else:
      raise
    return runcfg(cfgfile, badfilelist)
  print("all files left are good")

@contextlib.contextmanager
def cd(newdir):
  """http://stackoverflow.com/a/24176022/5228524"""
  prevdir = os.getcwd()
  os.chdir(os.path.expanduser(newdir))
  try:
    yield
  finally:
    os.chdir(prevdir)

def cdtemp(): return cd(tempfile.mkdtemp())

class KeepWhileOpenFile(object):
  def __init__(self, name, message=None):
    self.filename = name
    self.__message = message
    self.pwd = os.getcwd()
    self.fd = self.f = None
    self.bool = False

  @property
  def wouldbevalid(self):
    if self: return True
    with self:
      return bool(self)

  def __open(self):
    self.fd = os.open(self.filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

  def __enter__(self):
    with cd(self.pwd):
      try:
        self.__open()
      except OSError:
        return None

      self.f = os.fdopen(self.fd, 'w')

      try:
        if self.__message is not None:
          self.f.write(self.__message+"\n")
      except IOError:
        pass
      try:
        self.f.close()
      except IOError:
        pass
      self.bool = True
      return True

  def __exit__(self, *args):
    if self:
      try:
        with cd(self.pwd):
          os.remove(self.filename)
      except OSError:
        pass #ignore it
    self.fd = self.f = None
    self.bool = False

  def __nonzero__(self):
    return self.bool

class OneAtATime(KeepWhileOpenFile):
  def __init__(self, name, delay, message=None, printmessage=None, task="doing this"):
    super(OneAtATime, self).__init__(name, message=message)
    self.delay = delay
    if printmessage is None:
      printmessage = "Another process is already {task}!  Waiting {delay} seconds."
    printmessage = printmessage.format(delay=delay, task=task)
    self.__printmessage = printmessage

  def __enter__(self):
    while True:
      result = super(OneAtATime, self).__enter__()
      if result:
        return result
      print(self.__printmessage)
      time.sleep(self.delay)

if __name__ == "__main__":
  with cdtemp():
    shutil.copy(args.cfgfile, ".")

    badfilelist = args.badfilelist
    if badfilelist is None:
      badfilelist = os.path.join(os.path.dirname(cfgfile, "../../../run/DataFiles/baddatafiles.txt"))

    runcfg(os.path.basename(args.cfgfile), args.badfilelist)
