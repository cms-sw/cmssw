#!/usr/bin/env python3

import argparse
import contextlib
import errno
import glob
import os
import re
import shutil
import stat
import subprocess
import sys

thisfile = os.path.abspath(__file__)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("foldername", help="folder name for the campaign.  Example: CRUZET20xy")
  parser.add_argument("--cmssw", default=os.environ["CMSSW_VERSION"])
  parser.add_argument("--scram-arch", default=os.environ["SCRAM_ARCH"])
  parser.add_argument("--subfolder", default="", help="subfolder within basedir to make 'foldername' in.")
  parser.add_argument("--merge-topic", action="append", help="things to cms-merge-topic within the CMSSW release created", default=[])
  parser.add_argument("--print-sys-path", action="store_true", help=argparse.SUPPRESS) #internal, don't use this
  parser.add_argument('--basedir', default="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HipPy")
  args = parser.parse_args()

  basedir = args.basedir
  if not os.path.exists(basedir):
    raise FileExistsError("Base Directory does not exist!")

  if basedir[-1] == '/':
    basedir = basedir[:-1] #No trailing slashed allowed
  
  if args.print_sys_path:
    print(repr(sys.path))
    return

  folder = os.path.join(basedir, args.subfolder, args.foldername)

  mkdir_p(folder)

  with cd(folder):
    if not os.path.exists(args.cmssw):
      os.environ["SCRAM_ARCH"] = args.scram_arch
      subprocess.check_call(["scram", "p", "CMSSW", args.cmssw])
    with cd(args.cmssw):
       cmsenv()
       for _ in args.merge_topic:
         subprocess.check_call(["git", "cms-merge-topic", _])
       os.system("eval $(scram ru -sh) && scram b -j 10")  #my cmsenv function isn't quite good enough for scram b purposes.  Also, http://stackoverflow.com/a/38792806/5228524

       if os.path.exists("src/Alignment/HIPAlignmentAlgorithm"):
         HIPAlignmentAlgorithm = os.path.abspath("src/Alignment/HIPAlignmentAlgorithm")
       else:
         with cd(os.environ["CMSSW_RELEASE_BASE"]):
           HIPAlignmentAlgorithm = os.path.abspath("src/Alignment/HIPAlignmentAlgorithm")

       assert os.path.exists(HIPAlignmentAlgorithm), HIPAlignmentAlgorithm

    mkdir_p("Jobs")
    mkdir_p("run")

    with cd("run"):
      subprocess.check_call(["git", "init"])

      mkdir_p("Configurations")
      with cd("Configurations"):
        if not os.path.exists("align_tpl_py.txt"):
          shutil.copy(os.path.join(HIPAlignmentAlgorithm, "python", "align_tpl_py.txt"), ".")
          subprocess.check_call(["git", "add", "align_tpl_py.txt"])
        if not os.path.exists("common_cff_py_TEMPLATE.txt"):
          shutil.copy(os.path.join(HIPAlignmentAlgorithm, "python", "common_cff_py.txt"), "common_cff_py_TEMPLATE.txt")
          subprocess.check_call(["git", "add", "common_cff_py_TEMPLATE.txt"])
        mkdir_p("TrackSelection")
        with cd("TrackSelection"):
          for _ in glob.iglob(os.path.join(HIPAlignmentAlgorithm, "python", "*TrackSelection_cff_py.txt")):
            if not os.path.exists(os.path.basename(_)):
              shutil.copy(_, ".")
              subprocess.check_call(["git", "add", os.path.basename(_)])

      mkdir_p("DataFiles")
      with cd("DataFiles"):
        if not os.path.exists("data_example.lst"):
          with open("data_example.lst", "w") as f:
            f.write(os.path.join(os.getcwd(), "minbias.txt") + ",,MBVertex,Datatype:0\n")
            f.write(os.path.join(os.getcwd(), "cosmics.txt") + ",,COSMICS,Datatype:1 APVMode:deco Bfield:3.8T\n")
            f.write(os.path.join(os.getcwd(), "CDCs.txt") + ",,CDCS,Datatype:1 APVMode:deco Bfield:3.8T\n")
          subprocess.check_call(["git", "add", "data_example.lst"])
        if not os.path.exists("baddatafiles.txt"):
          with open("baddatafiles.txt", "w") as f:
            f.write("If any data files are bad (e.g. not at CERN), put them here,\n")
            f.write("separated by newlines or spaces or nothing or whatever you like.\n")
            f.write("Anything else in this file, like these lines, will be ignored.\n")
            f.write("You can also run hippyaddtobaddatafiles.py .../align_cfg.py to automatically\n")
            f.write("find bad data files.\n")
            f.write("Running jobs will automatically pick up changes here next time they resubmit.")

      mkdir_p("IOV")
      with cd("IOV"):
        if not os.path.exists("RunXXXXXX"):
          with open("RunXXXXXX", "w") as f:
            f.write("XXXXXX")
          subprocess.check_call(["git", "add", "RunXXXXXX"])

      if not os.path.exists("submit_template.sh"):
        shutil.copy(os.path.join(HIPAlignmentAlgorithm, "test", "hippysubmittertemplate.sh"), "submit_template.sh")
        os.chmod("submit_template.sh", os.stat("submit_template.sh").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        subprocess.check_call(["git", "add", "submit_template.sh"])
        
      if not os.path.exists("submit_script.sh"):
        shutil.copy(os.path.join(HIPAlignmentAlgorithm, "test", "hippysubmitterscript.sh"), "submit_script.sh")
        os.chmod("submit_script.sh", os.stat("submit_script.sh").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        subprocess.check_call(["git", "add", "submit_script.sh"])

      print("Dumped files into ", folder)
      
      try:
        subprocess.check_output(["git", "diff", "--staged", "--quiet"])
      except subprocess.CalledProcessError:
        subprocess.check_call(["git", "commit", "-m", "commit templates"])

def mkdir_p(path):
  """http://stackoverflow.com/a/600612/5228524"""
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

@contextlib.contextmanager
def cd(newdir):
  """http://stackoverflow.com/a/24176022/5228524"""
  prevdir = os.getcwd()
  os.chdir(os.path.expanduser(newdir))
  try:
    yield
  finally:
    os.chdir(prevdir)

def cmsenv():
  output = subprocess.check_output(["scram", "ru", "-sh"])
  for line in output.decode('utf8').split(";\n"):
    if not line.strip(): continue
    match1 = re.match(r'^export (\w*)="([^"]*)"$', line)
    match2 = re.match(r'^unset *((\w* *)*)$', line)
    if match1:
      variable, value = match1.groups()
      os.environ[variable] = value
    elif match2:
      for variable in match2.group(1).split():
        del os.environ[variable]
    else:
      raise ValueError("Bad scram ru -sh line:\n"+line)
  sys.path[:] = eval(subprocess.check_output([thisfile, "dummy", "--print-sys-path"]))

if __name__ == "__main__":
  main()
