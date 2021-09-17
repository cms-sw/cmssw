import urllib2, re, simplejson as json, socket
import sys

URL = "http://vocms00169:2113"
SQL = "select \
        r.RUNNUMBER,\
        r.START_TIME,\
        r.RUN_TRIGGERS,\
        r.RUN_TYPE\
      from \
        hcal.runs r \
      where \
        r.RUN_GLOBAL = 0 and \
        (r.runnumber >= :rf or :rf = 0) and \
        (r.runnumber <= :rt or :rt = 0) \
      order by \
        r.runnumber"

TIME = "select to_char(p.time,'YYYY-MM-DD HH24:MI:SS') time from hcal.run_parameters p where p.name like '%TRIGGERS'"

def query(query):
  resp = urllib2.urlopen(URL + "/query", query)
  if "getcode" in dir(resp) and resp.getcode() == 200:
    return resp.read()

def qstring(qstring):
  ps = ""
  for k in qstring.keys():
    ps += "&" if ps != "" else ""
    ps += "%s=%s" % (k, qstring[k])
  return ps

def get_single(q, qs):
  resp = urllib2.urlopen("%s/query/%s/data?%s" % (URL, query(q), qstring(qs)))
  if "getcode" in dir(resp) and resp.getcode() == 200:
    return json.loads(resp.read())["data"][0][0]
  return None

def get_all(q, qs):
  qid = query(q)
  ps = qstring(qs)
  u = "%s/query/%s/count?%s" % (URL, qid, ps)
  resp = urllib2.urlopen(u)
  if "getcode" in dir(resp) and resp.getcode() == 200:
    data = []
    c = int(resp.read())
    p = 1
    while c > 0:
      u = "%s/query/%s/page/1000/%d/data?%s" % (URL, qid, p, ps)
      resp = urllib2.urlopen(u)
      if "getcode" in dir(resp) and resp.getcode() == 200:
        j = json.loads(resp.read())
        data.extend(j["data"])
      p += 1
      c -= 1000
    return data
  return None

def main(rf, rt):
  ps = {}
  if rf != None: ps["rf"] = rf
  if rt != None: ps["rt"] = rt
  data = get_all(SQL, ps)
  for r in data:
    if r[3] != None:
      for t in ["pedestal","LED","laser"]:
        if re.search(t, r[3], flags=re.IGNORECASE) != None:
          d = get_single(TIME, { "p.run": int(r[0])}) 
          print r[0], t, "\"" + d + "\"", r[2]
          break
  

if __name__ == '__main__':
  
  rf = sys.argv[1] if len(sys.argv) > 1 else "-"
  rt = sys.argv[2] if len(sys.argv) > 2 else "-"
  rf = int(rf) if rf.isdigit() else 0
  rt = int(rt) if rt.isdigit() else 0

  main(rf, rt)
