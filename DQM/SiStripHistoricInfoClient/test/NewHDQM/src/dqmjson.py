from x509auth import *
from ROOT import TBufferFile, TH1F, TProfile, TH1F, TH2F
import re

X509CertAuth.ssl_key_file, X509CertAuth.ssl_cert_file = x509_params()
print X509CertAuth.ssl_key_file
print X509CertAuth.ssl_cert_file
print x509_params()

def dqm_get_json(server, run, dataset, folder, rootContent=False):
    postfix = "?rootcontent=1" if rootContent else ""
    datareq = urllib2.Request(('%s/data/json/archive/%s/%s/%s%s') % (server, run, dataset, folder, postfix))
    datareq.add_header('User-agent', ident)
    # Get data
    data = eval(re.sub(r"\bnan\b", "0", urllib2.build_opener(X509CertOpen()).open(datareq).read()),
               { "__builtins__": None }, {})
    if rootContent:
        # Now convert into real ROOT histograms   
        for idx,item in enumerate(data['contents']):
            if 'obj' in item.keys():
                if 'rootobj' in item.keys(): 
                    a = array('B')
                    a.fromstring(item['rootobj'].decode('hex'))
                    t = TBufferFile(TBufferFile.kRead, len(a), a, False)
                    rootType = item['properties']['type']
                    if rootType == 'TPROF': rootType = 'TProfile'
                    if rootType == 'TPROF2D': rootType = 'TProfile'
                    data['contents'][idx]['rootobj'] = t.ReadObject(eval(rootType+'.Class()'))
    return dict( [ (x['obj'], x) for x in data['contents'][1:] if 'obj' in x] )

def dqm_get_samples(server, match, type="offline_data"):
    datareq = urllib2.Request(('%s/data/json/samples?match=%s') % (server, match))
    datareq.add_header('User-agent', ident)
    # Get data
    data = eval(re.sub(r"\bnan\b", "0", urllib2.build_opener(X509CertOpen()).open(datareq).read()),
               { "__builtins__": None }, {})
    ret = []
    for l in data['samples']:
        if l['type'] == type:
            ret += [ (int(x['run']), x['dataset']) for x in l['items'] ]
    return ret
