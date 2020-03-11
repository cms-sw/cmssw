import re

# Various regular expressions used to check filename validity:

# Regexp for valid dataset names.
RXDATASET = re.compile(r"^(/[-A-Za-z0-9_]+){3}$")
# Regexp for valid RelVal dataset names.
RXRELVALMC = re.compile(r"^/RelVal[^/]+/(CMSSW(?:_[0-9]+)+(?:_pre[0-9]+)?)[-_].*$")
RXRELVALRUNDEPMC = re.compile(r"^/RelVal[^/]+/(CMSSW(?:_[0-9]+)+(?:_pre[0-9]+)?)[-_].*rundepMC.*$")
RXRELVALDATA = re.compile(r"^/[^/]+/(CMSSW(?:_[0-9]+)+(?:_pre[0-9]+)?)[-_].*$")
RXRUNDEPMC = re.compile(r"^/(?!RelVal)[^/]+/.*rundepMC.*$")

# Regexp for online DQM files.
RXONLINE = re.compile(r"^(?:.*/)?DQM_V(\d+)(_[A-Za-z0-9]+)?_R(\d+)\.root$")

# Regexp for offline DQM files.
RXOFFLINE = re.compile(r"^(?:.*/)?DQM_V(\d+)_R(\d+)((?:__[-A-Za-z0-9_]+){3})\.root$")

# --------------------------------------------------------------------
# Pre-classify a file into main category based on file name structure.
#   path: path (relative to the uploads dir, coming from the walk) of the root
#         file
# Returns a tuple of:
#   a boolean: True or False depending on whether the classification went OK
#   a string or dictionary:
#       - In case the classification went wrong: A string with the reason
#       - In case the classification was OK: A dictionary with classification
#                                            information
def classifyDQMFile(path):
  print(path)
  try:
    m = re.match(RXONLINE, path)
    if m:
      version = int(m.group(1))
      runnr = int(m.group(3))
      subsys = m.group(2) and m.group(2)[1:]
      if version != 1:
        return False, "file version is not 1"
      elif runnr <= 10000:
        return False, "online file has run number <= 10000"
      else:
        # online_data
        return True, { 'class': 'online_data', 'version': version,
                       'subsystem': subsys, 'runnr': runnr,
                       'dataset': "/Global/Online/ALL" }

    m = re.match(RXOFFLINE, path)
    if m:
      version = int(m.group(1))
      dataset = m.group(3).replace("__", "/")
      if not re.match(RXDATASET, dataset):
        return False, "Invalid dataset name"
      relvalmc = re.match(RXRELVALMC, dataset)
      relvaldata = re.match(RXRELVALDATA, dataset)
      relvalrundepmc = re.match(RXRELVALRUNDEPMC, dataset)
      rundepmc = re.match(RXRUNDEPMC, dataset)
      runnr = int(m.group(2))
      if version != 1:
        return False, "file version is not 1"
      if runnr < 1:
         return False, "file matches offline naming, but run number is < 1"
      elif rundepmc:
        if runnr == 1:
          return False,  "file matches Run Dependent MonteCarlo naming, but run number is 1"
        else:
          # simulated_rundep
          return True, { 'class': 'simulated_rundep', 'version': version,
                         'runnr': runnr, 'dataset': dataset }
      elif relvalrundepmc:
        if runnr == 1:
          return False, "file matches Run Dependent MonteCarlo naming, but run number is 1"
        else:
          # relval_rundepmc
          return True, { 'class': 'relval_rundepmc', 'version': version,
                         'runnr': runnr, 'dataset': dataset,
                         'release': relvalrundepmc.group(1)}
      elif relvalmc:
        if runnr != 1:
          return False, "file matches relval mc naming, but run number != 1"
        else:
          # relval_mc
          return True, { 'class': 'relval_mc', 'version': version,
                         'runnr': runnr, 'dataset': dataset,
                         'release': relvalmc.group(1) }
      elif relvaldata:
        if runnr == 1:
          return False, "file matches relval data naming, but run number = 1"
        else:
          # relval_data
          return True, { 'class': 'relval_data', 'version': version,
                         'runnr': runnr, 'dataset': dataset,
                         'release': relvaldata.group(1) }
      elif dataset.find("CMSSW") >= 0:
        return False, "non-relval dataset name contains 'CMSSW'"
      elif runnr > 1:
        # offline_data
        return True, { 'class': 'offline_data', 'version': version,
                       'runnr': runnr, 'dataset': dataset }
      else:
        # simulated
        return True, { 'class': 'simulated', 'version': int(m.group(1)),
                       'runnr': runnr, 'dataset': dataset }

    return False, "file matches no known naming convention"
  except:
    return False, "error while classifying file name"

