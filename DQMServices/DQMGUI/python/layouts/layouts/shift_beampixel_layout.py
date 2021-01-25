from .adapt_to_new_backend import *
dqmitems={}

def shiftbeamlayout(i, p, *rows): i["00 Shift/BeamPixel/" + p] = rows

shiftbeamlayout(dqmitems, "00 - Report Summary Map",
                [{ 'path': "BeamPixel/EventInfo/reportSummaryMap",
                   'description': "Pixel-Vertices Beam Spot: % Good Fits"}])

shiftbeamlayout(dqmitems, "A - fit results",
                [{ 'path': "BeamPixel/A - fit results",
                   'description': "Results of Beam Spot Fit"}])

shiftbeamlayout(dqmitems, "B - muX vs lumi",
                [{ 'path': "BeamPixel/B - muX vs lumi",
                   'description': "muX vs. Lumisection"}])

shiftbeamlayout(dqmitems, "B - muY vs lumi",
                [{ 'path': "BeamPixel/B - muY vs lumi",
                   'description': "muY vs. Lumisection"}])

shiftbeamlayout(dqmitems, "B - muZ vs lumi",
                [{ 'path': "BeamPixel/B - muZ vs lumi",
                   'description': "muZ vs. Lumisection"}])

shiftbeamlayout(dqmitems, "C - sigmaX vs lumi",
                [{ 'path': "BeamPixel/C - sigmaX vs lumi",
                   'description': "sigmaX vs. Lumisection"}])

shiftbeamlayout(dqmitems, "C - sigmaY vs lumi",
                [{ 'path': "BeamPixel/C - sigmaY vs lumi",
                   'description': "sigmaY vs. Lumisection"}])

shiftbeamlayout(dqmitems, "C - sigmaZ vs lumi",
                [{ 'path': "BeamPixel/C - sigmaZ vs lumi",
                   'description': "sigmaZ vs. Lumisection"}])

shiftbeamlayout(dqmitems, "G - vertex x fit",
                [{ 'path': "BeamPixel/G - vertex x fit",
                   'description': "Primary Vertex X Distribution (For Fit)"}])

shiftbeamlayout(dqmitems, "G - vertex y fit",
                [{ 'path': "BeamPixel/G - vertex y fit",
                   'description': "Primary Vertex Y Distribution (For Fit)"}])

shiftbeamlayout(dqmitems, "G - vertex z fit",
                [{ 'path': "BeamPixel/G - vertex z fit",
                   'description': "Primary Vertex Z Distribution (For Fit)"}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
