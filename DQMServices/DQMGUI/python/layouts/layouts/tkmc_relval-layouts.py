from .adapt_to_new_backend import *
dqmitems={}

def trackervalidationlayout(i, p, *rows): i["MCLayouts/Tk/" + p] = rows

trackervalidationlayout(dqmitems, "01 - TrackerDigisV",
                        [{ 'path':"TrackerDigisV/TrackerDigis/Pixel/adc_layer1ring1",
                           'description': ""},
                         {'path':"TrackerDigisV/TrackerDigis/Pixel/adc_layer1ring2",
                          'description': ""}],
                        [{'path':"TrackerDigisV/TrackerDigis/Strip/adc_tec_wheel1_ring1_zm",
                          'description': ""},
                         {'path':"TrackerDigisV/TrackerDigis/Strip/adc_tec_wheel1_ring1_zp",
                          'description': ""}])
                         
trackervalidationlayout(dqmitems, "02 - TrackerHitsV",
                        [{ 'path':"TrackerHitsV/TrackerHit/BPIXHit/Eloss_BPIX_1",
                           'description': ""},
                         {'path':"TrackerHitsV/TrackerHit/FPIXHit/Eloss_FPIX_1",
                          'description': ""}],
                        [{'path':"TrackerHitsV/TrackerHit/TECHit/Eloss_TEC_1",
                          'description': ""},
                         {'path':"TrackerHitsV/TrackerHit/TIBHit/Eloss_TIB_1",
                          'description': ""}])

trackervalidationlayout(dqmitems, "03 - TrackerRecHitsV",
                        [{ 'path':"TrackerRecHitsV/TrackerRecHits/Pixel/clustBPIX/Clust_charge_Layer1_Module1",
                           'description': ""},
                         {'path':"TrackerRecHitsV/TrackerRecHits/Pixel/clustFPIX/Clust_charge_Disk1_Plaquette1",
                          'description': ""}],
                        [{'path':"TrackerRecHitsV/TrackerRecHits/Strip/TEC/Adc_rphi_layer1tec",
                          'description': ""},
                         {'path':"TrackerRecHitsV/TrackerRecHits/Strip/TID/Adc_rphi_layer1tid",
                          'description': ""}])

trackervalidationlayout(dqmitems, "04 - TrackingMCTruth",
                        [{ 'path':"Tracking/TrackingMCTruth/TrackingParticle/TPAllHits",
                           'description': ""},
                         {'path':"Tracking/TrackingMCTruth/TrackingParticle/TPCharge",
                          'description': ""}],
                        [{'path':"Tracking/TrackingMCTruth/TrackingParticle/TPEta",
                          'description': ""},
                         {'path':"Tracking/TrackingMCTruth/TrackingParticle/TPPhi",
                          'description': ""}])


trackervalidationlayout(dqmitems, "05 - TrackingRecHits",
                        [{ 'path':"Tracking/TrackingRecHits/Pixel/Histograms_all/meChargeBarrel",
                           'description': ""},
                         {'path':"Tracking/TrackingRecHits/Pixel/Histograms_all/meChargeZmPanel1",
                          'description': ""}],
                        [{'path':"Tracking/TrackingRecHits/Strip/TEC/Adc_rphi_layer1tec",
                          'description': ""},
                         {'path':"Tracking/TrackingRecHits/Strip/TEC/Adc_rphi_layer2tec",
                          'description': ""}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
