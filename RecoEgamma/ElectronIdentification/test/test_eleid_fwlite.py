from RecoEgamma.ElectronIdentification.FWLite import electron_mvas, working_points
from DataFormats.FWLite import Events, Handle

# Small script to validate Electron MVA implementation in FWlite

import numpy as np
import pandas as pd

print('open input file...')

events = Events('root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_6_0/RelValZEE_13/'+ \
        'MINIAODSIM/PU25ns_106X_mcRun2_asymptotic_v3_FastSim-v1/10000/7BD68C01-3BA4-A74D-8B37-3EA162D34590.root')

# Get Handles on the electrons and other products needed to calculate the MVAs
ele_handle  = Handle('std::vector<pat::Electron>')
rho_handle  = Handle('double')
conv_handle = Handle('reco::ConversionCollection')
bs_handle   = Handle('reco::BeamSpot')

n = 100000

data = {"Fall17IsoV2"         : np.zeros(n),
        "Fall17IsoV2-wp80"    : np.zeros(n, dtype=bool),
        "Fall17IsoV2-wp90"    : np.zeros(n, dtype=bool),
        "Fall17IsoV2-wpLoose" : np.zeros(n, dtype=bool),
        "Fall17IsoV2-wpHZZ"   : np.zeros(n, dtype=bool),

        "Fall17NoIsoV2"         : np.zeros(n),
        "Fall17NoIsoV2-wp80"    : np.zeros(n, dtype=bool),
        "Fall17NoIsoV2-wp90"    : np.zeros(n, dtype=bool),
        "Fall17NoIsoV2-wpLoose" : np.zeros(n, dtype=bool),

        "Spring16HZZV1"         : np.zeros(n),
        "Spring16HZZV1-wpLoose" : np.zeros(n, dtype=bool),

        "Spring16GPV1"         : np.zeros(n),
        "Spring16GPV1-wp80"    : np.zeros(n, dtype=bool),
        "Spring16GPV1-wp90"    : np.zeros(n, dtype=bool),

        "nEvent"        : -np.ones(n, dtype=int),
        "pt"            : np.zeros(n)}

print('start processing')

accepted = 0
for i,event in enumerate(events): 

    nEvent = event._event.id().event()

    print("processing event {0}: {1}...".format(i, nEvent))

    # Save information on the first electron in an event,
    # if there is any the first electron of the

    event.getByLabel(('slimmedElectrons'), ele_handle)
    electrons = ele_handle.product()

    if not len(electrons):
        continue

    event.getByLabel(('fixedGridRhoFastjetAll'), rho_handle)

    rho = rho_handle.product()

    ele = electrons[0]
    i = accepted

    if ele.pt() in data["pt"][i-10:i]:
        continue

    data["nEvent"][i]           = nEvent
    data["pt"][i]               = ele.pt()

    mva, category = electron_mvas["Fall17IsoV2"](ele, rho)
    data["Fall17IsoV2"][i] = mva
    data["Fall17IsoV2-wp80"][i] = working_points["Fall17IsoV2"].passed(ele, mva, category, 'wp80')
    data["Fall17IsoV2-wp90"][i] = working_points["Fall17IsoV2"].passed(ele, mva, category, 'wp90')
    data["Fall17IsoV2-wpLoose"][i] = working_points["Fall17IsoV2"].passed(ele, mva, category, 'wpLoose')
    data["Fall17IsoV2-wpHZZ"][i] = working_points["Fall17IsoV2"].passed(ele, mva, category, 'wpHZZ')

    mva, category = electron_mvas["Fall17NoIsoV2"](ele, rho)
    data["Fall17NoIsoV2"][i] = mva
    data["Fall17NoIsoV2-wp80"][i] = working_points["Fall17NoIsoV2"].passed(ele, mva, category, 'wp80')
    data["Fall17NoIsoV2-wp90"][i] = working_points["Fall17NoIsoV2"].passed(ele, mva, category, 'wp90')
    data["Fall17NoIsoV2-wpLoose"][i] = working_points["Fall17NoIsoV2"].passed(ele, mva, category, 'wpLoose')

    mva, category = electron_mvas["Spring16HZZV1"](ele, rho)
    data["Spring16HZZV1"][i] = mva
    data["Spring16HZZV1-wpLoose"][i] = working_points["Spring16HZZV1"].passed(ele, mva, category, 'wpLoose')

    mva, category = electron_mvas["Spring16GPV1"](ele, rho)
    data["Spring16GPV1"][i] = mva
    data["Spring16GPV1-wp80"][i] = working_points["Spring16GPV1"].passed(ele, mva, category, 'wp80')
    data["Spring16GPV1-wp90"][i] = working_points["Spring16GPV1"].passed(ele, mva, category, 'wp90')

    accepted += 1

    if accepted==n:
        break

ele_df = pd.DataFrame(data)
ele_df = ele_df[ele_df["nEvent"] > 0]
ele_df.to_hdf("test_eleid_fwlite.h5", key="electron_data")
