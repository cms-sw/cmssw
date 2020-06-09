import asyncio
import random
import json
import time
import aiohttp

BASEURL = "http://localhost:7000"

CONNLIMIT = asyncio.Semaphore(10)

async def loadsamples(dataset, run):
    url = f"{BASEURL}/data/json/samples?match={dataset}&run={run}"
    async with CONNLIMIT:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                assert resp.status == 200, f"Server responded with error {resp.status} on request '{url}'"
                data = await resp.text()
                obj = json.loads(data)
                return [(it['dataset'], it['run']) for it in obj['samples'][0]['items']]

async def listdir(dataset, run, folder = ""):
    url = f"{BASEURL}/data/json/archive/{run}{dataset}/{folder}"
    async with CONNLIMIT:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                assert resp.status == 200, f"Server responded with error {resp.status} on request '{url}'"
                data = await resp.text()
                obj = json.loads(data)
                return ([(it['subdir']) for it in obj['contents'] if 'subdir' in it], 
                        [(it['obj']) for it in obj['contents'] if 'obj' in it])

async def getobject(dataset, run, name):
    url = f"{BASEURL}/plotfairy/archive/{run}{dataset}/{name}?w=266;h=200"
    start = time.time()
    async with CONNLIMIT:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                status = resp.status
                data = await resp.read()
                took = time.time() - start
                res = (status, took, len(data))
                assert resp.status == 200, f"Server responded with error {resp.status} on request '{url}'"
                return res
        
async def recursivelist(dataset, run, folder = ""):
    dirs, objs = await listdir(dataset, run, folder)
    return (len(await asyncio.gather(*[getobject(dataset, run, folder + "/" + name) for name in objs])) + 
            sum(await asyncio.gather(*[recursivelist(dataset, run, folder + "/" + sub) for sub in dirs])))



async def main():

    samp = await loadsamples("/", "")

    for s in samp:
        print(s)
        now = time.time()
        await listdir(*s)
        print(f"listdir: {time.time() - now:.3f}s")

        now = time.time()
        l = await recursivelist(*s, "")
        tot = time.time() - now
        print(f"recursivelist: {tot:.3f}s for {l} requests ({l/tot:.1f}/s)")
   
asyncio.run(main())
