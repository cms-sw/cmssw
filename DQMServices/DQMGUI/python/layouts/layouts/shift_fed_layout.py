from .adapt_to_new_backend import *
dqmitems={}

def fedlayout(i, p, *rows): i["00 Shift/FED/" + p] = rows

fedlayout(dqmitems, "00 Report Summary Map",
  [{ 'path': "FED/EventInfo/reportSummaryMap",
     'description': 'FED summary for each subdetector'}])

fedlayout(dqmitems, "01 FED Integrity Check",
  [{ 'path': "FED/FEDIntegrity_EvF/FedEntries",
     'description': 'Number of entries vs FED ID'}],
  [{ 'path': "FED/FEDIntegrity_EvF/FedFatal",
     'description': 'Number of fatal errors vs FED ID'}],
  [{ 'path': "FED/FEDIntegrity_EvF/FedNonFatal",
     'description': 'Number of non-fatal errors vs FED ID'}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
