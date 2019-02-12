from __future__ import print_function
from __future__ import absolute_import

import abc
import csv
import os
import re

import Utilities.General.cmssw_das_client as das_client

from .utilities import cache

class DatasetError(Exception): pass

defaultdasinstance = "prod/global"

class RunRange(object):
  def __init__(self, firstrun, lastrun, runs):
    self.firstrun = firstrun
    self.lastrun = lastrun
    self.runs = runs

  def __contains__(self, run):
    if self.runs and run not in self.runs: return False
    return self.firstrun <= run <= self.lastrun

def dasquery(dasQuery, dasLimit=0):
  dasData = das_client.get_data(dasQuery, dasLimit)
  if isinstance(dasData, str):
    jsondict = json.loads( dasData )
  else:
    jsondict = dasData
  # Check, if the DAS query fails
  try:
    error = findinjson(jsondict, "data","error")
  except KeyError:
    error = None
  if error or findinjson(jsondict, "status") != 'ok' or "data" not in jsondict:
    try:
      jsonstr = findinjson(jsondict, "reason")
    except KeyError: 
      jsonstr = str(jsondict)
    if len(jsonstr) > 10000:
      jsonfile = "das_query_output_%i.txt"
      i = 0
      while os.path.lexists(jsonfile % i):
        i += 1
      jsonfile = jsonfile % i
      theFile = open( jsonfile, "w" )
      theFile.write( jsonstr )
      theFile.close()
      msg = "The DAS query returned an error.  The output is very long, and has been stored in:\n" + jsonfile
    else:
      msg = "The DAS query returned a error.  Here is the output\n" + jsonstr
    msg += "\nIt's possible that this was a server error.  If so, it may work if you try again later"
    raise DatasetError(msg)
  return findinjson(jsondict, "data")

def getrunnumbersfromfile(filename, trydas=True, allowunknown=False, dasinstance=defaultdasinstance):
  parts = filename.split("/")
  error = None
  if parts[0] != "" or parts[1] != "store":
    error = "does not start with /store"
  elif parts[2] in ["mc", "relval"]:
    return [1]
  elif not parts[-1].endswith(".root"):
    error = "does not end with something.root"
  elif len(parts) != 12:
    error = "should be exactly 11 slashes counting the first one"
  else:
    runnumberparts = parts[-5:-2]
    if not all(len(part)==3 for part in runnumberparts):
      error = "the 3 directories {} do not have length 3 each".format("/".join(runnumberparts))
    try:
      return [int("".join(runnumberparts))]
    except ValueError:
      error = "the 3 directories {} do not form an integer".format("/".join(runnumberparts))

  if error and trydas:
    try:
      query = "run file={} instance={}".format(filename, dasinstance)
      dasoutput = dasquery(query)
      result = findinjson(dasoutput, "run")
      return sum((findinjson(run, "run_number") for run in result), [])
    except Exception as e:
      error = str(e)

  if error and allowunknown:
    return [-1]

  if error:
    error = "could not figure out which run number this file is from.\nMaybe try with allowunknown=True?\n  {}\n{}".format(filename, error)
    raise DatasetError(error)

def findinjson(jsondict, *strings):
  if len(strings) == 0:
    return jsondict
  if isinstance(jsondict,dict):
    if strings[0] in jsondict:
      try:
        return findinjson(jsondict[strings[0]], *strings[1:])
      except KeyError:
        pass
  else:
    for a in jsondict:
      if strings[0] in a:
        try:
          return findinjson(a[strings[0]], *strings[1:])
        except (TypeError, KeyError):  #TypeError because a could be a string and contain strings[0]
          pass
  #if it's not found
  raise KeyError("Can't find " + strings[0])

class DataFile(object):
  def __init__(self, filename, nevents, runs=None, trydas=True, allowunknown=False, dasinstance=defaultdasinstance):
    self.filename = filename
    self.nevents = int(nevents)
    if runs is None:
      runs = getrunnumbersfromfile(filename, trydas=trydas, allowunknown=allowunknown, dasinstance=dasinstance)
    if isinstance(runs, str):
      runs = runs.split()
    self.runs = [int(_) for _ in runs]

  def getdict(self):
    return {"filename": self.filename, "nevents": str(self.nevents), "runs": " ".join(str(_) for _ in self.runs)}

class DatasetBase(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def getfiles(self, usecache):
    pass

  @abc.abstractproperty
  def headercomment(self):
    pass

  def writefilelist_validation(self, firstrun, lastrun, runs, maxevents, outputfile=None, usecache=True):
    runrange = RunRange(firstrun=firstrun, lastrun=lastrun, runs=runs)

    if outputfile is None:
      outputfile = os.path.join(os.environ["CMSSW_BASE"], "src", "Alignment", "OfflineValidation", "python", self.filenamebase+"_cff.py")

    if maxevents < 0: maxevents = float("inf")
    totalevents = sum(datafile.nevents for datafile in self.getfiles(usecache) if all(run in runrange for run in datafile.runs))
    if totalevents == 0:
      raise ValueError("No events within the run range!")
    accepted = rejected = 0.  #float so fractions are easier

    fractiontoaccept = 1.*maxevents / totalevents

    with open(outputfile, "w") as f:
      f.write("#"+self.headercomment+"\n")
      f.write(validationheader)
      for datafile in self.getfiles(usecache):
        if all(run in runrange for run in datafile.runs):
          if accepted == 0 or accepted / (accepted+rejected) <= fractiontoaccept:
            f.write('"' + datafile.filename + '",\n')
            accepted += datafile.nevents
          else:
            rejected += datafile.nevents
        elif any(run in runrange for run in datafile.runs):
          raise DatasetError("file {} has multiple runs {}, which straddle firstrun or lastrun".format(datafile.filename, datafile.runs))
      f.write("#total events in these files: {}".format(accepted))
      f.write(validationfooter)

  def writefilelist_hippy(self, firstrun, lastrun, runs, eventsperjob, maxevents, outputfile, usecache=True):
    runrange = RunRange(firstrun=firstrun, lastrun=lastrun, runs=runs)
    if maxevents < 0: maxevents = float("inf")
    totalevents = sum(datafile.nevents for datafile in self.getfiles(usecache) if all(run in runrange for run in datafile.runs))
    if totalevents == 0:
      raise ValueError("No events within the run range!")
    accepted = rejected = inthisjob = 0.  #float so fractions are easier

    fractiontoaccept = 1.*maxevents / totalevents
    writecomma = False

    with open(outputfile, "w") as f:
      for datafile in self.getfiles(usecache):
        if all(run in runrange for run in datafile.runs):
          if accepted == 0 or accepted / (accepted+rejected) <= fractiontoaccept:
            if writecomma: f.write(",")
            f.write("'" + datafile.filename + "'")
            accepted += datafile.nevents
            inthisjob += datafile.nevents
            if inthisjob >= eventsperjob:
              f.write("\n")
              inthisjob = 0
              writecomma = False
            else:
              writecomma = True
          else:
            rejected += datafile.nevents
        elif any(run in runrange for run in datafile.runs):
          raise DatasetError("file {} has multiple runs {}, which straddle firstrun or lastrun".format(datafile.filename, datafile.runs))
      f.write("\n")

class Dataset(DatasetBase):
  def __init__(self, datasetname, dasinstance=defaultdasinstance):
    self.datasetname = datasetname
    if re.match(r'/.+/.+/.+', datasetname):
      self.official = True
      self.filenamebase = "Dataset" + self.datasetname.replace("/","_")
    else:
      self.official = False
      self.filenamebase = datasetname

    self.dasinstance = dasinstance

  @cache
  def getfiles(self, usecache):
    filename = os.path.join(os.environ["CMSSW_BASE"], "src", "Alignment", "CommonAlignment", "data", self.filenamebase+".csv")
    if not usecache:
      try:
        os.remove(filename)
      except OSError as e:
        if os.path.exists(filename):
          raise

    result = []
    try:
      with open(filename) as f:
        for row in csv.DictReader(f):
          result.append(DataFile(**row))
        return result
    except IOError:
      pass

    query = "file dataset={} instance={} detail=true | grep file.name, file.nevents".format(self.datasetname, self.dasinstance)
    dasoutput = dasquery(query)
    if not dasoutput:
      raise DatasetError("No files are available for the dataset '{}'. This can be "
                         "due to a typo or due to a DAS problem. Please check the "
                         "spelling of the dataset and/or try again.".format(datasetname))
    result = [DataFile(findinjson(_, "file", "name"), findinjson(_, "file", "nevents")) for _ in dasoutput if int(findinjson(_, "file", "nevents"))]
    try:
      with open(filename, "w") as f:
        writer = csv.DictWriter(f, ("filename", "nevents", "runs"))
        writer.writeheader()
        for datafile in result:
          writer.writerow(datafile.getdict())
    except Exception as e:
      print("Couldn't write the dataset csv file:\n\n{}".format(e))
    return result

  @property
  def headercomment(self):
    return self.datasetname

class MultipleDatasets(DatasetBase):
  def __init__(self, *datasets, **kwargs):
    dasinstance = defaultdasinstance
    for kw, kwarg in kwargs.iteritems():
      if kw == "dasinstance":
        dasinstance = kwarg
      else:
        raise TypeError("Unknown kwarg {}={}".format(kw, kwarg))
    self.datasets = [Dataset(dataset, dasinstance=dasinstance) for dataset in datasets]

  @cache
  def getfiles(self, usecache):
    return sum([d.getfiles(usecache=usecache) for d in self.datasets], [])

  @property
  def headercomment(self):
    return ", ".join(d.headercomment for d in self.datasets)

validationheader = """
import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
"""

validationfooter = """
] )
"""
