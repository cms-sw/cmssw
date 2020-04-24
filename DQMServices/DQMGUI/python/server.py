import re
import os
import json
import time
import traceback
import http.server

from urllib.parse import parse_qs
from urllib.parse import urlparse

BASE = os.getenv("CMSSW_BASE") + "/src/DQMServices/DQMGUI/data/"
# to get the frontend files, and also the DB there.
os.chdir(BASE)

from DQMServices.DQMGUI.render import RenderPool
import DQMServices.DQMGUI.rootstorage as storage

renderpool = RenderPool(workers=5)

if len(storage.searchsamples()) == 0:
    import glob
    EOSPATH = "/eos/cms/store/group/comm_dqm/DQMGUI_data/Run*/*/R000*/DQM_*.root"
    EOSPREFIX = "root://eoscms.cern.ch/"
    print(f"Listing all files on EOS ({EOSPATH}), this can take a while...")
    files = glob.glob(EOSPATH)
    storage.registerfiles([EOSPREFIX + f for f in files])
    print(f"Done, registered {len(files)} files.")

def samples(args):
    args = parse_qs(args)
    match = args['match'][0] if 'match' in args else None
    run = args['run'][0] if 'run' in args else None
    items = []
    structure = {'samples': [
        {'type':'dqmio_data', 'items': items }]}

    for s in storage.searchsamples(match, run):
        item = {'type':'dqmio_data', 'run': s.run, 'dataset': s.dataset, 'version':''}
        items.append(item)
    return structure

def list(run, dataset, folder):
    run = int(run)
    sample = storage.Sample(dataset, run, 0, None, None, None)
    if len(folder) > 1 and not folder.endswith("/"):
        folder += "/"
    items = storage.listmes(sample, folder.encode("utf-8"), recursive = False)
    contents = [{ "streamerinfo":"" }]
    structure = {"contents": contents}
    for itembytes in sorted(items):
        item = itembytes.decode("utf-8")
        if item[-1] == "/":
            contents.append({'subdir': item[:-1]})
        else:
            contents.append({"obj": item, "properties": {}})
    return structure

def jsroot(run, dataset, fullname, args):
    run = int(run)
    mes = reader.readsampleme(dataset, run, 0, fullname)
    accu = mes.pop().data
    tlist = ROOT.TList()
    for me in mes:
        tlist.Add(me.data)
    accu.Merge(tlist)
    return str(ROOT.TBufferJSON.ConvertToJSON(accu)).encode("utf-8")

def plotpng(run, dataset, fullname, args):
    run = int(run)
    args = parse_qs(args)
    width = int(args['w'][0]) if 'w' in args else 400
    height = int(args['h'][0]) if 'h' in args else 320
    #while width < 300 or height < 200:
    #    width, height = width * 2, height * 2
    if 'w' in args: del args['w']
    if 'h' in args: del args['h']
    spec = ";".join(k + "=" + v[0] for k, v in args.items())
    sample = storage.Sample(dataset, run, 0, None, None, None)
    mes = storage.readme(sample, fullname.encode("utf-8"))
    if mes:
        obj = mes[0]
        if isinstance(obj, bytes):
            # this is a normal histogram
            effi = [x for x in mes if isinstance(x, storage.EfficiencyFlag)]
            with renderpool.renderer() as r:
                png, error = r.renderhisto(obj, [], name = fullname, spec=spec, width=width, height=height, efficiency = bool(effi))
                if error:
                    # maybe we are missing streamers, retry with file.
                    s = storage.queryforsample(sample)
                    png, error = r.renderhisto(obj, [], name = fullname, spec=spec, width=width, height=height, 
                                               efficiency = bool(effi), streamerfile = s.filename.encode("utf-8"))
        elif isinstance(obj, storage.ScalarValue):
            with renderpool.renderer() as r:
                png, error = r.renderscalar(obj.value.decode("utf-8"),  width=width, height=height)
        else:
            with renderpool.renderer() as r:
                png, error = r.renderscalar("object not found", width=width, height=height)
            
        if error:
            print("Rendering error for ", run, dataset, fullname, args)
        return png

def index():
    datasets = sorted(set(x.dataset for x in storage.searchsamples()))
    return (['<ul>'] +
        [f'<li><a href="/runsfordataset/{ds}">{ds}</a></li>' for ds in datasets] +
        ['</ul>'])

def listruns(dataset):
    runs = sorted(set(x.run for x in storage.searchsamples(dataset)))
    return ([f'<h1>{dataset}</h1><ul>'] +
        [f'<li><a href="/showdata/{run}{dataset}/">{run}</a></li>' for run in runs] +
        ['</ul>'])

def showdata(run, dataset, folder):
    run = int(run)
    sample = storage.Sample(dataset, run, 0, None, None, None)
    items = sorted(storage.listmes(sample, folder.encode("utf-8"), recursive = False))
    out = [f'<h1><a href="/runsfordataset/{dataset}">{dataset}</a> {run}</h1>', '<h2>']
    breadcrumbs = folder.split("/")
    for i in range(len(breadcrumbs)-1):
        part = "/".join(breadcrumbs[:i+1]) + "/"
        out.append(f'<a href="/showdata/{run}{dataset}/{part}"> {breadcrumbs[i]} </a>/')
    out.append("</h2><ul>")
    for itembytes in items:
        item = itembytes.decode("utf-8")
        if item[-1] == '/':
            out.append(f'<li><a href="/showdata/{run}{dataset}/{folder}{item}">{item}</a></li>')
    out.append("</ul>")
    for itembytes in items:
        item = itembytes.decode("utf-8")
        if item[-1] != '/':
            out.append(f'<a href="/plotfairy/archive/{run}{dataset}/{folder}{item}?w=1000&h=800"><img src="/plotfairy/archive/{run}{dataset}/{folder}{item}" /></a>')
    return out


ROUTES = [
    (re.compile('/$'), index, "html"),
    (re.compile('/runsfordataset/(.*)$'), listruns, "html"),
    (re.compile('/showdata/([0-9]+)(/[^/]+/[^/]+/[^/]+)/(.*)'), showdata, "html"),
    (re.compile('/+data/json/samples[?]?(.+)?'), samples, "json"),
    (re.compile('/+data/json/archive/([0-9]+)(/[^/]+/[^/]+/[^/]+)/?(.*)'), list, "json"),
    (re.compile('/+jsrootfairy/archive/([0-9]+)(/[^/]+/[^/]+/[^/]+)/([^?]*)[?]?(.+)?'), jsroot, "application/json"),
    (re.compile('/+plotfairy/archive/([0-9]+)(/[^/]+/[^/]+/[^/]+)/([^?]*)[?]?(.+)?'), plotpng, "image/png"),
]


# the server boilerplate.
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsedParams = urlparse(self.path)
        if os.access("./" + parsedParams.path.replace("..", ""), os.R_OK):
            return http.server.SimpleHTTPRequestHandler.do_GET(self);
        try:
            res = None
            for pattern, func, type in ROUTES:
                m = pattern.match(self.path)
                if m:
                    res = func(*m.groups())
                    break

            if res:
                self.send_response(200, "Here you go")
                if type == "html":
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b"""<html><style>
                        body {
                            font-family: sans;
                        }
                    </style><body>""")
                    self.wfile.write("\n".join(res).encode("utf-8"))
                    self.wfile.write(b"</body></html>")
                elif type == "json":
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(res).encode("utf-8"))
                else:
                    self.send_header("Content-Type", f"{type}; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(res)
            else:
                self.send_response(400, "Something went wrong")
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"I don't understand this request.")
        except:
            trace = traceback.format_exc()
            self.send_response(500, "Things went very wrong")
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(trace.encode("utf8"))

PORT = 8889
server_address = ('', PORT)
httpd = http.server.ThreadingHTTPServer(server_address, Handler)
print(f"Serving at http://localhost:{PORT}/ ...")
httpd.serve_forever()
