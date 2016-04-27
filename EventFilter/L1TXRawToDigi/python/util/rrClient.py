import re, json, datetime
from rrapi import RRApi, RRApiError

URL  = "http://runregistry.web.cern.ch/runregistry/"

def getRunsNewer(run, minLumis):
    try:
        api = RRApi(URL)

        if api.app == "user":
            result = api.data(workspace = 'GLOBAL',
                    table = 'runsummary',
                    template = 'json',
                    columns = [
                            'number',
                            'lsCount',
                            'startTime', 'duration',
                            'hltkey', 'gtKey', 'l1Menu', 'tscKey', 'triggerMode',
                            'triggers',
                            'runClassName',
                        ],
                    query = "{number} > %d and {lsCount} > %d and {triggers} > 0" % (run, minLumis),
                )
            runs = {}
            for runDict in result :
                runNo = int(runDict['number'])
                runDict['date'] = datetime.datetime.strptime(runDict['startTime'], "%a %d-%m-%y %H:%M:%S").date().strftime('%Y%m%d')
                runs[runNo] = runDict
            return runs

        else :
            print "RunRegistry API 'app' != user, who knows why... :<"

    except RRApiError, e:
        print e

if __name__ == '__main__' :
    # Test run
    print json.dumps(getRunsNewer(250400, 10), indent=4)
