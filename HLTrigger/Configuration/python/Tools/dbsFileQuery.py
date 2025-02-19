def dbsFileQuery(query):
  # load the DBS API
  import sys, os
  dbsapipath = os.environ['DBSCMD_HOME']
  if dbsapipath not in sys.path:
    sys.path.append(dbsapipath)

  from dbsApi import DbsApi
  api = DbsApi()

  # run the query, which should be of the form 'find file where ...'
  results = api.executeQuery(query)

  # parse the results in XML format, and extract the list of files
  from xml.dom.minidom import parseString
  xml = parseString(results)

  files = []

  for results in xml.getElementsByTagName('results'):
    for row in results.getElementsByTagName('row'):
      for file in row.getElementsByTagName('file'):
        files.append( file.firstChild.data )

  return files
