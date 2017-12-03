import ROOT

# from within CMSSW:
ROOT.gSystem.Load('libCondFormatsBTauObjects') 
ROOT.gSystem.Load('libCondToolsBTau') 


# get the sf data loaded
calib = ROOT.BTagCalibration('csvv1', 'CSVV1.csv')

# making a std::vector<std::string>> in python is a bit awkward, 
# but works with root (needed to load other sys types):
#v_sys = getattr(ROOT, 'vector<string>')()
#v_sys.push_back('up')
#v_sys.push_back('down')

# make a reader instance and load the sf data
reader = ROOT.BTagCalibrationReader(
    0,              # 0 is for loose op, 1: medium, 2: tight, 3: discr. reshaping
    "central",      # central systematic type
)    
reader.load(
    calib, 
    0,          # 0 is for b flavour, 1: FLAV_C, 2: FLAV_UDSG 
    "comb"      # measurement type
)
# reader.load(...)     # for FLAV_C
# reader.load(...)     # for FLAV_UDSG

# in your event loop
sf = reader.eval_auto_bounds(
    'central',      # systematic (here also 'up'/'down' possible)
    0,              # jet flavor
    2.5,            # eta
    31.             # pt
)
print sf

sf1 = reader.eval_auto_bounds(
    'central',      # systematic (here also 'up'/'down' possible)
    0,              # jet flavor
    2.4,            # eta
    25.             # pt
)
print sf1

sf2 = reader.eval_auto_bounds(
    'central',      # systematic (here also 'up'/'down' possible)
    0,              # jet flavor
    2.3,            # eta
    31.             # pt
)
print sf2

sf3 = reader.eval_auto_bounds(
    'central',      # systematic (here also 'up'/'down' possible)
    0,              # jet flavor
    1.3,            # eta
    21.             # pt
)
print sf3


