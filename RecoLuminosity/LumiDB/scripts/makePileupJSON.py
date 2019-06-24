#!/usr/bin/env python
from __future__ import print_function
import argparse
import RecoLuminosity.LumiDB.LumiConstants as LumiConstants
import re
from math import sqrt
import six

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

# This script takes a .csv file containing the per-BX luminosity values from brilcalc and processes them to
# produce an output file containing the average and RMS pileup for each lumi section. This can then be fed
# into pileupCalc.py to calculate a final pileup histogram. For more documentation see:
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData

# modified from the estimatePileup.py script in RecoLuminosity/LumiDB
# originally 5 Jan, 2012  Mike Hildreth

if __name__ == '__main__':
    parameters = LumiConstants.ParametersObject()

    parser = argparse.ArgumentParser(description="Script to estimate average and RMS pileup using the per-bunch luminosity information provided by brilcalc. The output is a JSON file containing a dictionary by runs with one entry per lumi section.")
    parser.add_argument('inputFile', help='CSV input file as produced from brilcalc')
    parser.add_argument('outputFile', help='Name of JSON output file')
    parser.add_argument('-b', '--selBX', metavar='BXLIST', help='Comma-separated list of BXs to use (will use all by default)')
    parser.add_argument('-n', '--no-threshold', action='store_true', help='By default, to avoid including spurious luminosity from afterglow in the pileup calculation, bunches with luminosity below a given threshold are excluded. This threshold is 8.0/ub/LS for HFOC at nominal energy, 2.0/ub/LS for HFOC at low energy, and 1.2/ub/LS for other luminometers. If the input data has already been preprocessed (e.g. by using the --xingTr argument in brilcalc) to exclude these bunches, or if you are running on special data with low overall luminosity, then use this flag to disable the application of the threshold.')
    args = parser.parse_args()

    output = args.outputFile

    sel_bx = set()
    if args.selBX:
        for ibx in args.selBX.split(","):
            try:
                bx = int(ibx)
                sel_bx.add(bx)
            except:
                print(ibx,"is not an int")
        print("Processing",args.inputFile,"with selected BXs:", sorted(sel_bx))
    else:
        print("Processing",args.inputFile,"with all BX")

    # The "CSV" file actually is a little complicated, since we also want to split on the colons separating
    # run/fill as well as the spaces separating the per-BX information.
    sepRE = re.compile(r'[\]\[\s,;:]+')
    csv_input = open(args.inputFile, 'r')
    last_run = -1

    last_valid_lumi = []

    output_line = '{'
    for line in csv_input:
        if line[0] == '#':
            continue # skip comment lines

        pieces = sepRE.split(line.strip())

        if len(pieces) < 15:
            # The most likely cause of this is that we're using a csv file without the bunch data, so might as well
            # just give up now.
            raise RuntimeError("Not enough fields in input line; maybe you forgot to include --xing in your brilcalc command?\n"+line)
        try:
            run = int(pieces[0])
            lumi_section = int(pieces[2])
            beam_energy = float(pieces[10])
            #tot_del_lumi = float(pieces[11])
            #tot_rec_lumi = float(pieces[12])
            luminometer = pieces[14]

            if luminometer == "HFOC":
                if beam_energy > 3000:
                    # Use higher threshold for nominal runs otherwise we'll get a lot of junk.
                    threshold = 8.0
                else:
                    # Use lower threshold for 5.02 GeV runs since the overall lumi is quite low.
                    threshold = 2.0
            else:
                threshold = 1.2

            xing_lumi_array = []
            for bxid, bunch_del_lumi, bunch_rec_lumi in zip(pieces[15::3], pieces[16::3], pieces[17::3]):
                if sel_bx and int(bxid) not in sel_bx:
                    continue
                if args.no_threshold or float(bunch_del_lumi) > threshold:
                    xing_lumi_array.append([int(bxid), float(bunch_del_lumi), float(bunch_rec_lumi)])
        except:
            print("Failed to parse line: check if the input format has changed")
            print(pieces[0],pieces[1],pieces[2],pieces[3],pieces[4],pieces[5],pieces[6],pieces[7],pieces[8],pieces[9])
            continue

        # In principle we could also have a check for if len(pieces) == 15 (i.e. no BX data is present) but
        # luminosity is present, which implies we're using a luminometer without BX data. In this case we
        # could just extrapolate from the previous good LS (just scaling the pileup by the ratio of
        # luminosity). In practice this is an extremely small number of lumi sections, and in 2018 the 14
        # lumisections in normtag_PHYSICS without BX data (all RAMSES) are all in periods with zero recorded
        # lumi, so they don't affect the resulting pileup at all. So for run 2 I haven't bothered to implement
        # this.

        if run != last_run:
            # the script also used to add a dummy LS at the end of runs but this is not necessary in run 2
            if last_run > 0:
                output_line = output_line[:-1] + '], '
            last_run = run
            output_line += ('\n"%d":' % run )
            output_line += ' ['

        # Now do the actual parsing.
        total_lumi = 0 
        total_int = 0
        total_int2 = 0
        total_weight2 = 0

        # first loop to get sum for (weighted) mean
        for bxid, bunch_del_lumi, bunch_rec_lumi in xing_lumi_array:
            total_lumi += bunch_rec_lumi
            # this will eventually be_pileup*bunch_rec_lumi but it's quicker to apply the factor once outside the loop
            total_int += bunch_del_lumi*bunch_rec_lumi
        # filled_xings = len(xing_lumi_array)
        
        # convert sum to pileup and get the mean
        total_int *= parameters.orbitLength / parameters.lumiSectionLength
        if total_lumi > 0:
            mean_int = total_int/total_lumi
        else:
            mean_int = 0

        # second loop to get (weighted) RMS
        for bxid, bunch_del_lumi, bunch_rec_lumi in xing_lumi_array:
            mean_pileup = bunch_del_lumi * parameters.orbitLength / parameters.lumiSectionLength
            if mean_pileup > 100:
                print("mean number of pileup events > 100 for run %d, lum %d : m %f l %f" % \
                      (runNumber, lumi_section, mean_pileup, bunch_del_lumi))
                #print "mean number of pileup events for lum %d: m %f idx %d l %f" % (lumi_section, mean_pileup, bxid, bunch_rec_lumi)

            total_int2 += bunch_rec_lumi*(mean_pileup-mean_int)*(mean_pileup-mean_int)
            total_weight2 += bunch_rec_lumi*bunch_rec_lumi

        # compute final RMS and write it out
        #print " LS, Total lumi, filled xings %d, %f, %d" %(lumi_section,total_lumi,filled_xings)
        bunch_rms_lumi = 0
        denom = total_lumi*total_lumi-total_weight2
        if total_lumi > 0 and denom > 0:
            bunch_rms_lumi = sqrt(total_lumi*total_int2/denom)

        output_line += "[%d,%2.4e,%2.4e,%2.4e]," % (lumi_section, total_lumi, bunch_rms_lumi, mean_int)
        last_valid_lumi = [lumi_section, total_lumi, bunch_rms_lumi, mean_int]
        
    output_line = output_line[:-1] + ']}'
    csv_input.close()

    outputfile = open(output,'w')
    if not outputfile:
        raise RuntimeError("Could not open '%s' as an output JSON file" % output)

    outputfile.write(output_line)
    outputfile.close()
    print("Output written to", output)
