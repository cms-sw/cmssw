import json
from subprocess import check_output
from sys import argv

def dasql(x):
    txtjson = check_output(["das_client.py", "--query=%s" %x, "--limit=10000", "--format=json"])
    return json.loads(txtjson)

def go(dataset):
    try:
        dsets = dasql("dataset dataset=%s" % dataset)['data']
    except:
        print "Failed DAS query for %s" % dataset
        return
    if "*" in dataset:
        for d in dsets:
            go(d['dataset'][0]['name'])
        return
    for d in dsets:
        for record in d['dataset']:
            if 'mcm' in record:
                mcm = record['mcm']
                gen = mcm['generator_parameters'][0]
                print mcm['dataset_name'], gen['cross_section'], gen['filter_efficiency'], mcm['completed_events']

if __name__ == "__main__":
    go(argv[1])
