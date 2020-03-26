DQMIO Python Libraries
======================

This package provides PyROOT based Python code to read DQMIO files.

This library is designed for Python3. Use a `PY3` CMSSW IB to get a Python3-compatible PyROOT.

Usage
-----

Very basic: Read local files.

```
> from DQMServices.FwkIO.DQMIO import DQMIOReader
> r = DQMIOReader()
> # Add some local DQMIO data files
> r.addfiles("mydataset", ["/data/mschneid/179BF42D-D53E-3C45-92A5-DB2101E7E9FD.root"])
> # Read the metadata. This may take a while.
> r.checkfiles()

> # Now do some queries:
> r.samples()
[('mydataset', 315270, 53),
 ('mydataset', 315270, 54),
 ('mydataset', 315270, 55),
 ('mydataset', 315270, 0),
 ('mydataset', 315322, 540),
 ...
]
> r.listsample('mydataset', 315339, 0, "")
{'AlCaReco/',
 'AlcaBeamMonitor/',
 'CTPPS/',
 'Castor/',
 'DQM/',
 'DT/',
 'Ecal/',
 'EcalBarrel/',
 'EcalEndcap/',
 'EcalPreshower/',
 'Egamma/',
 'HLT/',
 'Hcal/',
 'Info/',
 'L1T/',
 'Muons/',
 'RPC/',
 'RecoTauV/',
 'SiStrip/',
 'Tracking/'}
> menames = r.filtersample('mydataset', 315339, 43, ".*DigiPhase1Task.*")
> menames
{'Hcal/DigiPhase1Task/ADC/SubdetPM/HBM',
 'Hcal/DigiPhase1Task/ADC/SubdetPM/HBP',
 'Hcal/DigiPhase1Task/ADC/SubdetPM/HEM',
 'Hcal/DigiPhase1Task/ADC/SubdetPM/HEP',
 ...
}
> mes = r.readsampleme('mydataset', 315339, 0, menames)
[MonitorElement(run=315339, lumi=0, name='Hcal/DigiPhase1Task/ADC/SubdetPM/HBM', type=3, data=<ROOT.TH1F object ("HBM") at 0x7fd040061460>),
 MonitorElement(run=315339, lumi=0, name='Hcal/DigiPhase1Task/ADC/SubdetPM/HBP', type=3, data=<ROOT.TH1F object ("HBM") at 0x7fd0180737e0>),
 MonitorElement(run=315339, lumi=0, name='Hcal/DigiPhase1Task/ADC/SubdetPM/HEM', type=3, data=<ROOT.TH1F object ("HFP") at 0x7fcff801a370>),
 MonitorElement(run=315339, lumi=0, name='Hcal/DigiPhase1Task/ADC/SubdetPM/HEP', type=3, data=<ROOT.TH1F object ("HEP") at 0x7fd0040cd370>),
 ...
]
```

More advanced: Read a dataset from DAS (do `voms-proxy-init -voms cms -rfc` first):
```
> r = DQMIOReader()
> r.importdataset("/EGamma/Run2018A-12Nov2019_UL2018-v2/DQMIO") 
> r.checkfiles() # this will take a while...
> r.readsampleme('/EGamma/Run2018A-12Nov2019_UL2018-v2/DQMIO', 315339, 39, "Hcal/DigiPhase1Task/Occupancy/depth/depth1")
[MonitorElement(run=315339, lumi=39, name='Hcal/DigiPhase1Task/Occupancy/depth/depth1', type=6, data=<ROOT.TH2F object ("depth1") at 0x7f5d2c068170>)]
> # Since this data is not harvested, this will return multiple MEs that need to be added to get the full run histogram.
> # For this reason, a ME will also only appear either in the per-lumi, or the per-run sections, but not both.
> # It will also be quite slow, since many files need to be read.
> r.readsampleme('/EGamma/Run2018A-12Nov2019_UL2018-v2/DQMIO', 315339, 0, "Hcal/DigiPhase1Task/ADC/SubdetPM/HEP")
[MonitorElement(run=315339, lumi=0, name='Hcal/DigiPhase1Task/ADC/SubdetPM/HEP', type=3, data=<ROOT.TH1F object ("HEP") at 0x7f5d74929110>),
 MonitorElement(run=315339, lumi=0, name='Hcal/DigiPhase1Task/ADC/SubdetPM/HEP', type=3, data=<ROOT.TH1F object ("HEP") at 0x7f5d749e08b0>),
 ...
] 
```

Most advanced: Read a lot of data using a database.

```
> r = DQMIOReader("/data/mschneid/egamma2018.db", 100) # persistent database and 100 IO threads
> r.importdatasets("/EGamma/Run2018*-12Nov2019_UL2018-v*/DQMIO")
> r.datasets()
['/EGamma/Run2018A-12Nov2019_UL2018-v2/DQMIO',
 '/EGamma/Run2018B-12Nov2019_UL2018-v2/DQMIO',
 '/EGamma/Run2018C-12Nov2019_UL2018-v2/DQMIO',
 '/EGamma/Run2018D-12Nov2019_UL2018-v3/DQMIO',
 '/EGamma/Run2018D-12Nov2019_UL2018-v4/DQMIO']
> r.checkfiles() # This will take a long time.
> # Actually, only 500 files are readable at the moment.
> list(r.db.execute("select count(*), readable, indexed from file group by readable, indexed")) 
[(4894, 0, None), (500, 1, 1)]
> # these are from the 2018B and C eras:
> set(s[0] for s in r.samples())
{'/EGamma/Run2018B-12Nov2019_UL2018-v2/DQMIO',
 '/EGamma/Run2018C-12Nov2019_UL2018-v2/DQMIO'}
```

Now we can use the existing database to read some MEs:

```
from DQMServices.FwkIO.DQMIO import DQMIOReader
r = DQMIOReader("/data/mschneid/egamma2018.db")
# Read a ME for all available lumis of a run. This will take a while.
mes = r.readlumimes('/EGamma/Run2018B-12Nov2019_UL2018-v2/DQMIO', 317649, 'EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 2')
print(mes)
```

By default, all operations print progress indications, since all operations might be quite slow (on remote files). However, on local files and/or with warm caches, this is may slow dwon the operations; use `DQMIOReader(..., progress=False) to disable the progress display on `read*` operations.

Implementation
--------------

The DQMIO format consists of ROOT `TTree`s with ME names and objects. The trees are sorted by name, and an additional tree withmetadata provides run/lumi information. This library implements binary search over the trees to be able to locate MEs in the DQMIO with a minimal amount of bytes read. There is caching and multi-threading to make this as efficient as possible even on remote files.

The code is split in two parts: The first implements all the low-level operations on DQMIO files, while the second provides the `DQMIOReader` class which handles a metadata database and multi-threading.

The low-level interface is built around `IndexEntry`s, the rows of the `Indices` metadata tree in the DQMIO file. This data is read from the datafiles and cached in a local SQLite database, the (potentially) remote ROOT files are only accessed to read MEs. There is a big cache (100's of files) of open `TFile`s, to avoid the long latency of locating remote files. Another cache holds values of the `FullName` column, so most of the binary search can happen without actually touching ROOT.

The high-level interface is built around a rather elaborate database to track file status: First, it is checked if a file is currently readable (on disk), then, it's metadata is extracted. Non-readable files will be re-checked every day. Interrupting and resuming `checkfiles()` should be no problem.
