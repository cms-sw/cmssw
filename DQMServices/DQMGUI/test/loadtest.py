import gzip
import time
import requests
from multiprocessing.pool import ThreadPool
# Make this file by extracting logs from the old GUI:
# ssh vocms0738 'unzip -p /data/srv/logs/dqmgui/offline/old-logs-20180514-0007.zip "*weblog*" | gzip' > /tmp/logs.gz
f = gzip.open('logs.gz','rb')

def plots():
    for line in f:
        if b'plotfairy' in line:
            #if not b'AIgfsb' in line: continue
            #if not b'overlay' in line: continue
            if b'unknown' in line: continue
            path = line.split(b" ")[4]
            path = path.split(b"plotfairy/")[1]
            path = path.replace(b"archive/", b"render/")
            path = path.replace(b"overlay?", b"render_overlay?")
            url = "http://shitbox2.cern.ch:1234/api/v1/" + path.decode("utf-8")
            yield url

gen = plots()
pool = ThreadPool(20)

# skip some requests from the beginning, as needed
[next(gen) for _ in range(100000)]

total = 0
while True:
    urls = [next(gen) for _ in range(1000)]
    start = time.time()
    resp = pool.map(requests.get, urls)
    good = 0
    for r in resp:
        if r.status_code == 200:
            good += 1
        else:
            pass
    total += 1000
    print(total, "good:", good, "in", time.time()-start)
