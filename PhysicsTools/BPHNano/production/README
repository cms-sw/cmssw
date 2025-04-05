#Multicrab production code

Largely inspired by G. Karahtanasis crab submission script

```
usage: submit_on_crab.py [-h] [-y YAML] [-c {submit,status}] [-f FILTER] [-w WORKAREA] [-o OUTPUTDIR] [-t TAG] [-p PSETCFG] [-e [EXTRA ...]]

A multicrab submission script

optional arguments:
  -h, --help            show this help message and exit
  -y YAML, --yaml YAML  File with dataset descriptions
  -c {submit,status}, --cmd {submit,status}
                        Crab command
  -f FILTER, --filter FILTER
                        filter samples, POSIX regular expressions allowed
  -w WORKAREA, --workarea WORKAREA
                        Crab working area name
  -o OUTPUTDIR, --outputdir OUTPUTDIR
                        LFN Output high-level directory: the LFN will be saved in outputdir+workarea
  -t TAG, --tag TAG     Production Tag extra
  -p PSETCFG, --psetcfg PSETCFG
                        Plugin configuration file
  -e [EXTRA ...], --extra [EXTRA ...]
                        Optional extra input files
  -tt, --test           Flag a test job
```

##Format of the yaml file with datasets

It is checked automatically, so if you see a yaml format error please keep in mind that the following format is used:

```
expected_schema = Schema({
    "common": {
        "data": {
            "lumimask": And(str, error="lumimask should be a string"),
            "splitting": And(int, error="splitting should be an integer"),
            "globaltag": And(str, error="globaltag should be a string"),
        },
        "mc": {
            "splitting": And(int, error="splitting should be an integer"),
            "globaltag": And(str, error="globaltag should be a string"),
        },
    },
    "samples": And(dict, error="samples should be a dict with keys dataset (str), isMC (bool). Optional keys: globaltag (str), parts (list(int))")
    }
    )

samples_schema = Schema({
    "dataset": And(str, error="dataset should be a string"),
    "isMC": And(bool, error="isMC should be a boolean"),
    Optional("decay") : And(str, error="decay to reconstruct"),
    Optional("goldenjson") : And(str, error="golden json file path should be a string"),
    Optional("globaltag") : And(str, error="globaltag should be a string"),
    Optional("parts"): [And(int, error="parts should be a list of integers")]
})
```



