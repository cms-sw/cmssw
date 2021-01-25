from .adapt_to_new_backend import *
dqmitems={}

def aliOfflinePVLayout(i, p, *rows): i["OfflinePV/AlignmentValidation/" + p] = rows 

aliOfflinePVLayout(dqmitems, "00 - Vertex and vertex tracks quality",
                   [{ 'path': "OfflinePV/Alignment/chi2ndf",
                      'description': "Chi square of vertex tracks (pT>1GeV)",
                      'draw': { 'withref': "no" }
                      },
                    { 'path': "OfflinePV/Alignment/chi2prob",
                      'description': "Chi square probability of vertex tracks (pT>1GeV)",
                      'draw': { 'withref': "no" }
                      }],
                   [{ 'path': "OfflinePV/Alignment/sumpt",
                      'description': "sum of transverse momentum squared of vertex tracks (pT>1GeV)",
                      'draw': { 'withref': "no" }
                      },
                    { 'path': "OfflinePV/Alignment/weight",
                      'description': "weight of track in vertex fit",
                      'draw': { 'withref': "no" }
                      }],
                   [{ 'path': "OfflinePV/Alignment/ntracks",
                      'description': "number of tracks in vertex",
                      'draw': { 'withref': "no" }
                      },
                    { 'path': "OfflinePV/Alignment/dxy",
                      'description': "transverse impact parameter",
                      'draw': { 'withref': "no" }
                      }])

aliOfflinePVLayout(dqmitems, "01 - Impact parameters and errors",
                   [{ 'path': "OfflinePV/Alignment/dxyzoom",
                      'description': "transverse impact parameter w.r.t vertex",
                      'draw': { 'withref': "no" }
                      },
                    { 'path': "OfflinePV/Alignment/dxyErr",
                      'description': "error on transverse impact parameter w.r.t vertex",
                      'draw': { 'withref': "no" }
                      }],
                   [{ 'path': "OfflinePV/Alignment/dz",
                      'description': "longitudinal impact parameter w.r.t vertex",
                      'draw': { 'withref': "no" }
                      },
                    { 'path': "OfflinePV/Alignment/dzErr",
                      'description': "error on longitudinal impact parameter w.r.t vertex",
                      'draw': { 'withref': "no" }
                      }])

aliOfflinePVLayout(dqmitems, "02 - Impact parameters projections (pT>1 GeV)",
                   [{ 'path': "OfflinePV/Alignment/dxyVsPhi_pt1",
                      'description': "transverse impact parameter vs track azimuth (track momentum > 1 GeV)",
                      'draw': { 'withref': "no" }
                      },
                    { 'path': "OfflinePV/Alignment/dxyVsEta_pt1",
                      'description': "transverse impact parameter vs track pseudorapitidy (track momentum > 1 GeV)",
                      'draw': { 'withref': "no" }
                      }],
                   [{ 'path': "OfflinePV/Alignment/dzVsPhi_pt1",
                      'description': "longitudinal impact parameter vs track azimuth (track momentum > 1 GeV)",
                      'draw': { 'withref': "no" }
                      },
                    { 'path': "OfflinePV/Alignment/dzVsEta_pt1",
                      'description': "longidutinal impact parameter vs track pseudorapidity (track momentum > 1 GeV)",
                      'draw': { 'withref': "no" }
                      }])

aliOfflinePVLayout(dqmitems, "03 - Impact parameters projections (pT>10 GeV)",
                   [{ 'path': "OfflinePV/Alignment/dxyVsPhi_pt10",
                      'description': "transverse impact parameter vs track azimuth (track momentum > 10 GeV)",
                      'draw': { 'withref': "no" }
                      },
                    { 'path': "OfflinePV/Alignment/dxyVsEta_pt10",
                      'description': "transverse impact parameter vs track pseudorapitidy (track momentum > 10 GeV)",
                      'draw': { 'withref': "no" }
                      }],
                   [{ 'path': "OfflinePV/Alignment/dzVsPhi_pt10",
                      'description': "longitudinal impact parameter vs track azimuth (track momentum > 10 GeV)",
                      'draw': { 'withref': "no" }
                      },
                    { 'path': "OfflinePV/Alignment/dzVsEta_pt10",
                      'description': "longidutinal impact parameter vs track pseudorapidity (track momentum > 10 GeV)",
                      'draw': { 'withref': "no" }
                      }])




apply_dqm_items_to_new_back_end(dqmitems, __file__)
