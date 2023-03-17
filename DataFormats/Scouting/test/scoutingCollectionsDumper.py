#!/usr/bin/env python3
import os
import argparse
import glob
import fnmatch
import ROOT

from DataFormats.FWLite import Runs, Events, Handle

def getScoutingProductsList(key):
    '''Returns list of tuples (type, label) for EDM Scouting collections
       - 'key' must be "Scouting" (data formats before Run 3), or "Run3Scouting"
    '''
    if key == 'Scouting':
        return [
            ('vector<ScoutingCaloJet>', 'hltScoutingCaloPacker'),
            ('vector<ScoutingElectron>', 'hltScoutingEgammaPacker'),
            ('vector<ScoutingMuon>', 'hltScoutingMuonPackerCalo'),
            ('vector<ScoutingPFJet>', 'hltScoutingPFPacker'),
            ('vector<ScoutingParticle>', 'hltScoutingPFPacker'),
            ('vector<ScoutingPhoton>', 'hltScoutingEgammaPacker'),
            ('vector<ScoutingTrack>', 'hltScoutingTrackPacker'),
            ('vector<ScoutingVertex>', 'hltScoutingPrimaryVertexPacker:primaryVtx'),
        ]
    elif key == 'Run3Scouting':
        return [
            ('vector<Run3ScoutingElectron>', 'hltScoutingEgammaPacker'),
            ('vector<Run3ScoutingMuon>', 'hltScoutingMuonPacker'),
            ('vector<Run3ScoutingPFJet>', 'hltScoutingPFPacker'),
            ('vector<Run3ScoutingParticle>', 'hltScoutingPFPacker'),
            ('vector<Run3ScoutingPhoton>', 'hltScoutingEgammaPacker'),
            ('vector<Run3ScoutingTrack>', 'hltScoutingTrackPacker'),
            ('vector<Run3ScoutingVertex>', 'hltScoutingMuonPacker:displacedVtx'),
            ('vector<Run3ScoutingVertex>', 'hltScoutingPrimaryVertexPacker:primaryVtx')
        ]
    else:
        raise RuntimeError(f'getScoutingProductsList -- invalid key (must be "Scouting", or "Run3Scouting"): "{key}"')

def printScoutingVar(name, value):
    '''Print content of data member of Scouting object
    '''
    if isinstance(value, ROOT.Run3ScoutingHitPatternPOD):
        for subvar in [
            'hitCount',
            'beginTrackHits',
            'endTrackHits',
            'beginInner',
            'endInner',
            'beginOuter',
            'endOuter',
            'hitPattern',
        ]:
            subvalue = getattr(value, subvar)
            print(f'      {name}.{subvar} = {subvalue}')
    else:
        print(f'      {name} = {value}')

def printScoutingProduct(product_label, product_type, product, verbosity):
    '''Print content of EDM product
    '''
    if verbosity == 0:
        return

    productIsVector = product_type.startswith('vector')

    productInfoStr = f'Product Label: "{product_label}" (type: "{product_type}")'
    if productIsVector:
        productInfoStr += f', size = {product.size()}'

    print(f'\n  {productInfoStr}')

    if not productIsVector:
        printScoutingVar('value', product[0])
        return

    obj_idx = 0
    for obj in product:
        # print only first N objects, where N corresponds to verbosity (if positive)
        if verbosity > 0 and obj_idx >= verbosity:
            break

        # names of data members to print
        if obj_idx == 0:
            varNames = sorted([foo for foo in dir(obj) if not fnmatch.fnmatch(foo, '__*__')])

        print(f'\n    Object #{obj_idx}')
        obj_idx += 1
        for varName in varNames:
            varValue = getattr(obj, varName)()
            printScoutingVar(varName, varValue)

def analyseEvent(event, productList, verbosity = -1):
    '''Function to analyse a single EDM Event
    '''
    if verbosity != 0:
        print('-'*50)
        print(f'Run             = {event.eventAuxiliary().run()}')
        print(f'LuminosityBlock = {event.eventAuxiliary().luminosityBlock()}')
        print(f'Event           = {event.eventAuxiliary().event()}')

    for productType, productLabel in productList:
        productHandle = Handle(productType)
        event.getByLabel(productLabel, productHandle)
        if productHandle.isValid():
            printScoutingProduct(productLabel, productType, productHandle.product(), verbosity)

    if verbosity != 0:
        print('-'*50)

def getInputFiles(inputList):
    '''List of input files (after resolving wildcards, removing duplicates, and sorting)
    '''
    ret = set()
    for input_i in inputList:
        inputFiles_i = glob.glob(input_i)
        if len(inputFiles_i) == 0:
            inputFiles_i = [input_i]
        for input_j in inputFiles_i:
            ret.add(os.path.abspath(os.path.realpath(input_j)) if os.path.isfile(input_j) else input_j)
    return sorted(list(ret))

###
### main
###
if __name__ == '__main__':
    ## args
    parser = argparse.ArgumentParser(
        description = 'FWLite script to print to stdout content of Scouting collections in EDM files.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i', '--inputs', dest='inputs', required=True, nargs='+', default=None,
                        help='List of EDM files in ROOT format')

    parser.add_argument('-s', '--skipEvents', dest='skipEvents', action='store', type=int, default=0,
                        help='Index of first event to be processed (inclusive)')

    parser.add_argument('-n', '--maxEvents', dest='maxEvents', action='store', type=int, default=-1,
                        help='Maximum number of events to be processed (inclusive)')

    parser.add_argument('-k', '--key', dest='key', action='store', type=str, choices=['Scouting', 'Run3Scouting'], default='Scouting',
                        help='Keyword to select Scouting DataFormats (must be "Scouting", or "Run3Scouting")')

    parser.add_argument('-v', '--verbosity', dest='verbosity', action='store', type=int, default=-1,
                        help='Level of verbosity')

    opts, opts_unknown = parser.parse_known_args()

    log_prx = os.path.basename(__file__)+' --'

    ## args validation
    if len(opts_unknown) > 0:
        raise RuntimeError(f'{log_prx} unrecognized command-line arguments: {opts_unknown}')

    inputFiles = getInputFiles(opts.inputs)

    if len(inputFiles) == 0:
        raise RuntimeError(f'{log_prx} empty list of input files [-i]')

    ## Event Loop
    nEvtRead, nEvtProcessed = 0, 0
    skipEvents = max(0, opts.skipEvents)

    scoutingProductsList = getScoutingProductsList(opts.key)

    for input_file in inputFiles:
        try:
            events = Events(input_file)
            for event in events:
                nEvtRead += 1
                if (nEvtRead <= skipEvents) or ((opts.maxEvents >= 0) and (nEvtProcessed >= opts.maxEvents)):
                    continue

                analyseEvent(event = event, productList = scoutingProductsList, verbosity = opts.verbosity)
                nEvtProcessed += 1

        except:
            print(f'{log_prx} failed to analyse TFile (file will be ignored): {input_file}')
            continue

    if opts.verbosity != 0:
        print(f'Events processed = {nEvtProcessed}')
