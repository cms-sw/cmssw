def fedlayout(i, p, *rows): i["00 Shift/FED/" + p] = DQMItem(layout=rows)

fedlayout(dqmitems, "00 FED Integrity Check",
  [{ 'path': "FED/FEDIntegrity_EvF/FedEntries",
     'description': 'Number of entries vs FED ID'}],
  [{ 'path': "FED/FEDIntegrity_EvF/FedFatal",
     'description': 'Number of fatal errors vs FED ID'}],
  [{ 'path': "FED/FEDIntegrity_EvF/FedNonFatal",
     'description': 'Number of non-fatal errors vs FED ID'}])
