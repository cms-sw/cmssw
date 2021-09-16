#!/usr/bin/env python3

from contentValuesLib import *
import xml.dom.minidom

reDatasetParts = "^/([^/]+)/([^/]+)/([^/]+)$"
DEFAULT_BASE = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/"

REMINDERS = [
  ('^/StreamExpress/', 'Express/%(numberPart1)03d/%(numberPart2)03d/DQM_V0001_R%(number)09d__StreamExpress__%(datasetPart2)s__%(datasetPart3)s.root'),
  ('^/Global/Online/ALL$', 'Online/%(numberPart1)03d/%(numberPart2)03d/DQM_V0003_R%(number)09d.root')
]

class OptionParser(optparse.OptionParser):
  """ Option parser class """
  def __init__(self):
    optparse.OptionParser.__init__(self, usage="%prog [options]", version="%prog 0.0.1", conflict_handler="resolve")
    self.add_option("--url", action="store", type="string", dest="url", default=SERVER_URL, help="specify RR XML-RPC server URL. Default is " + SERVER_URL)
    self.add_option("--from", action="store", type="int", dest="from", default=None, help="specify run number lower threshold inclusive. Note that all runs/datasets retrieval can take a long time!")
    self.add_option("--to", action="store", type="int", dest="to", default=None, help="specify run number upper threshold inclusive. Note that all runs/datasets retrieval can take a long time!")
    self.add_option("--base", action="store", type="string", dest="base", default=DEFAULT_BASE, help="file base. Default is " + DEFAULT_BASE)

def getNodeText(nodelist):
  rc = ""
  for node in nodelist:
    if node.nodeType == node.TEXT_NODE:
      rc = rc + node.data
  return rc

if __name__ == "__main__":
  
  # Create option parser and get options/arguments
  optManager  = OptionParser()
  (opts, args) = optManager.parse_args()
  opts = opts.__dict__

  server = xmlrpclib.ServerProxy(opts['url'])
  query = {'hasSummaryValues': 'false'}
  if opts['from'] != None:
    query['number'] = '>= ' + str(opts['from'])
  if opts['to'] != None:
    if 'number' in query:
      query['number'] += ' and <= ' + str(opts['to'])
    else:
      query['number'] = '<= ' + str(opts['to'])

  xmldata = server.DataExporter.export('GLOBAL', 'xml_all', query)

  datasets = []
  dom = xml.dom.minidom.parseString(xmldata)

  for run in dom.getElementsByTagName("RUN"):
    number = int(getNodeText(run.getElementsByTagName("NUMBER")[0].childNodes))
    for dataset in run.getElementsByTagName("DATASET"):
      datasets.append((number, getNodeText(dataset.getElementsByTagName("NAME")[0].childNodes)))

  for (number, dataset) in datasets:
    parts = None
    m = re.search(reDatasetParts, dataset)
    if m == None:
      sys.stderr.write("Wrong dataset name (run %d, dataset %s)!\n" % (number, dataset))
    else:
      parts = m.group(1, 2, 3)
      fullPath = None
      for (reminderPattern, reminderFormat) in REMINDERS:
        if re.match(reminderPattern, dataset):
          reminder = reminderFormat % \
                    { 'number': number, 'dataset': dataset, 
                      'numberPart1': int(re.search('^([0-9]+)([0-9]{3})$', str(number)).group(1)),
                      'numberPart2': int(re.search('^([0-9]+)([0-9]{3})$', str(number)).group(2)),
                      'datasetPart1': parts[0], 
                      'datasetPart2': parts[1], 
                      'datasetPart3': parts[2] 
                    }
          fullPath = opts['base'] + reminder
          break

      if fullPath == None:
        sys.stderr.write("Dataset was not identified (run %d, dataset %s)!\n" % (number, dataset))
      else:
        try:
          os.stat(fullPath)
          sys.stdout.write(fullPath) 
          sys.stdout.write('\n') 
        except:
          sys.stderr.write("File [%s] not exists or is not accessible (run %d, dataset %s)!\n" % (fullPath, number, dataset))


  sys.exit(0)


