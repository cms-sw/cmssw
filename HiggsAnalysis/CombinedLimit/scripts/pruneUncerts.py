#!/usr/bin/env python

from optparse import OptionParser, OptionGroup

## set up the option parser
parser = OptionParser(usage="usage: %prog [options] ARG",
                      description="This is a script to prune uncertainties of an existing set of datacards. The script needs as inputs the directory that contains all datacards, which are supposed to be processed and the output of the maximum likelihood fit and the script diffNuisances.py in text format. The latter should be passed on by option --fit-results. Note that option --fit-results can also take a list of these files produced on subsets of the datacards in question. In this case for correlated uncertainties the pruning decision will be based on the maximal shift in the individual fits. The script optionally takes a list of regular expressions for uncertainties that should not be considered for pruning using the option --blacklist. The pruning takes place based on the relative shift of the nuisance parameter in the maximum likelihood fit(s). The metric can be switched between the result of the background-only fit ('b'), the signal-plus-background fit ('s+b') or the maximum of the two ('max'). A threshold on the shift can be given via option --threshold. Each nuisance parameter, for which the relative shift in the according metric falls below this threshold will be be added to a list of uncertainties to be pruned. You can optionally choose to have these uncertainties already commented in the tested datacards at the same time. The output of the script is the list of pruned and a list of kept nuisance parameters in text format (uncertainty-pruning-[drop/keep].txt). In the current implementation the script is meant to be used for datacards for counting experiments or for shape analyses based on histograms. ARG corresponds to the input directory where to find the datacards.")
parser.add_option("--fit-results", dest="fit_results", default="", type="string",
                  help="The absolute path to the output file(s) of the maximum likelihood fit and the script diffNuisances.py, which returns the output of the maximum likelihood fit in txt format. For this purpose the script diffNuisances.py should be run with options -A, -f text and -a. Note that you can also pass a list of maximum likelihood fit outputs based on a subsets of datacards. In this case the absolute paths to all maximum likelihood fit output files should be passed on embraced by quotation marks and separated by whitespace. [Default: \"\"]")
parser.add_option("--metric", dest="metric", default="max",  type="choice", choices=['b', 's+b', 'max'],
                  help="The metric to be used for the pruning decision. Choices are: b (pull for background-only fit), s+b (pull for signal-plus-background fit), the maximum of 'b' and 's+b'. [Default: 'max']")
parser.add_option("-m", "--mass", dest="mass", default="125", type="string",
                  help="The mass value to be chosen, when considering the signal-plus-background fit in the metric. If the metric of the background-only fit is chosen this option has no effect. [Default: \"125\"]")
parser.add_option("--threshold", dest="threshold", default="0.05", type="string",
                  help="The threshold to determine the nuisance parameters to be pruned. The value corresponds to the relative shift of the parameter in the maximum likelihood fit(s). If the shift of the nuisance parameter falls below this threshold the nuisance parameter will be added to the list of nuisance parameters to be pruned. [Default: 0.05]")
parser.add_option("--blacklist", dest="blacklist", default="", type="string",
                  help="A list of regular python style expressions for nuisance parameters that should not be considered during the pruning decision. (You can add holy cows here.) The regular expressions should be embraced by quotation marks and separated by whitespace. [Default: \"\"]")
parser.add_option("--comment-nuisances", dest="comment_nuisances", default=False, action="store_true",
                  help="Use this option if you want to comment the uncertainties added to the list of pruned nuisance parameters from the tested datacards at the same time. [Default: False]")
(options, args) = parser.parse_args()
## check number of arguments; in case print usage
if len(args) < 1 :
    parser.print_usage()
    exit(1)

import os
import re
import glob
import math
import ROOT
import random
import string

def combine_fit_results(FIT_RESULTS, METRIC) :
    """
    Create a pseudo file of fit results form a list of files of fit result based on a subset of 
    datacards. From multiply occuring uncertainties the larger pulls will replace smaller pulls
    according to the corresponding METRIC. The combined pseudo file will be written to the tmp
    directory.
     - FIT_RESULTS is a list of fit result files based on a subset of datacards.
     - METRIC corresponds to the chosen metric. Available options are 'b', 's+b', 'max'.
    Return value will be the full path to the combined pseudo file. 
    """
    output = {}
    headline = ''
    pull_pattern = re.compile('[+-]\d+\.\d+(?=sig)')
    for fit_result in FIT_RESULTS :
        file= open(fit_result,'r')
        for line in file :
            ## add headline
            if 'name' in line :
                if headline == '' :
                    headline = line
                    continue
                else :
                    continue
            ## fill outputs with uncertainties of this channel
            key = line.split()[0]
            ## skip POI
            if key == 'r' :
                continue
            if not key in output.keys() :
                output[key] = line
            else :
                pulls_old  = pull_pattern.findall(output[key])
                pulls_new  = pull_pattern.findall(line)
                if pulls_new :
                    if METRIC == 'b' :
                        value_old = float(pulls_old[0])
                        value_new = float(pulls_new[0])
                    if METRIC == 's+b' :
                        value_old = float(pulls_old[1])
                        value_new = float(pulls_new[1])
                    if METRIC == 'max' :
                        value_old = max(abs(float(pulls_old[0])), float(pulls_old[1]))
                        value_new = max(abs(float(pulls_new[0])), float(pulls_new[1]))                        
                    if value_new > value_old :
                        output[key] = line
        file.close()
    rnd_name=''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(10))
    file= open("/tmp/{NAME}".format(NAME=rnd_name),'w')
    file.write(headline)
    for line in output.values() :
        file.write(line)
    file.close()
    return "/tmp/{NAME}".format(NAME=rnd_name)

def determine_shapes(DATACARD, MASS):
    """
    Determine all shape uncertainties from a given datacard. For all shape uncertainties the
    largest relative uncertainty over all bins is determined and added to a dictionary:
     - DATACARD is the absolute path to the datacard.
     - MASS     is the mass to be chosen in case signal samples are involved.
    Return value is a dictionary mapping the uncertainty name to the maximal relative uncert.
    """
    def valid(u_value, s_value, excepts) :
        """
        check for a weak equality between u_value and s_value. The argument excepts corresponds
        to exceptional cases for which '*' is not true. This function is used for proc and bin.
        """
        if u_value in excepts :
            return u_value == s_value
        else :
            return u_value == s_value or s_value == '*'
        
    ## determine list of bin names, list of process names and list of  input
    ## files which contain shapes
    bins = []
    procs = []
    shapes = []
    uncerts = []
    bin_excepts = []
    proc_excepts = []
    shape_uncerts = {}
    file = open(DATACARD, 'r')
    for line in file :
        words = line.split()
        if words[0] == 'bin' :
            if len(words[1:]) > len(bins) :
                bins = words[1:]
        if words[0] == 'process' :
            if not words[1].isdigit() :
                procs = words[1:]
        if words[0] == 'shapes' :
            if len(words) > 5 :
                if not words[2] == '*' :
                    bin_excepts.append(words[2])
                if not words[1] == '*' :
                    proc_excepts.append(words[1])                    
                ## ----------  proc      bin       path      shape     syst
                shapes.append((words[1], words[2], words[3], words[4], words[5]))
    file.close()
    ## determine shape uncertainties; for shape uncertainties it must be known
    ## for what bin and for what sample they are valid and what value they have
    file= open(DATACARD,'r')
    for line in file :
        words = line.split()
        if len(words)<2 :
            continue
        if words[1] == 'shape' :
            for idx in range(len(words[2:])) :
                if not words[idx+2] == '-' :
                    ## -----------  unc       proc        bins       value
                    uncerts.append((words[0], procs[idx], bins[idx], words[idx+2]))
    file.close()
    ## open root input files, find all shape histograms
    for (u_unc, u_proc, u_bin, u_value) in uncerts :
        for (s_proc, s_bin, s_path, s_shape, s_syst) in shapes :
            if valid(u_bin, s_bin, bin_excepts) :
                if valid(u_proc, s_proc, proc_excepts) :
                    root_file = ROOT.TFile(s_path, 'READ')
                    value = root_file.Get(s_shape.replace('$CHANNEL', u_bin).replace('$PROCESS', u_proc).replace('$MASS', MASS))
                    upper = root_file.Get(s_syst.replace('$CHANNEL', u_bin).replace('$PROCESS', u_proc).replace('$MASS', MASS).replace('$SYSTEMATIC', u_unc)+'Up')
                    lower = root_file.Get(s_syst.replace('$CHANNEL', u_bin).replace('$PROCESS', u_proc).replace('$MASS', MASS).replace('$SYSTEMATIC', u_unc)+'Down')
                    unc_max = 0
                    ## iterate over bins and find maximal relative difference
                    for i in range(value.GetNbinsX()) :
                        if value.GetBinContent(i+1) :
                            unc_bin = float(u_value)*(upper.GetBinContent(i+1)/value.GetBinContent(i+1)-1.)
                            if unc_max < unc_bin :
                                unc_max = unc_bin
                            unc_bin = float(u_value)*(1.-lower.GetBinContent(i+1)/value.GetBinContent(i+1))
                            if unc_max < unc_bin :
                                unc_max = unc_bin
                    root_file.Close()
                    shape_uncerts[u_unc] = unc_max
    return shape_uncerts

def determine_lnNs(DATACARD):
    """
    Determine all lnN uncertainties from a given datacard. For correlated uncertainties the
    largest relative uncertainty over all occurences is determined and added to a dictionary:
     - DATACARD is the absolute path to the datacard.
    Return value is a dictionary mapping the uncertainty name to the maximal relative uncert.
    """
    lnN_uncerts = {}
    file= open(DATACARD,'r')
    for line in file :
        words = line.split()
        if len(words)<2 :
            continue
        if words[1] == 'lnN' :
            unc_max = 0
            for idx in range(len(words[2:])) :
                if not words[idx+2] == '-' :
                    if unc_max < float(words[idx+2]) :
                        unc_max = float(words[idx+2])
            lnN_uncerts[words[0]] = unc_max
    file.close()
    return lnN_uncerts

def determine_uncerts(DATACARD, MASS) :
    """
    Determine full list of uncertainties as a dictionary mapping uncertainty name to the maximal
    relative uncertainty. Uncertainties of type shape and lnN are supported.
     - DATACARD is the absolute path to the datacard.
    Return value is a dictionary mapping the uncertainty name to the maximal relative uncert.    
    """
    uncerts = determine_lnNs(DATACARD)
    uncerts.update(determine_shapes(DATACARD, MASS))
    return uncerts

def prune(FITRESULT, UNCERTS, METRIC, THRESHOLD, BLACKLIST) :
    """
    Take the pruning decision by relative shift of the uncertainty by the maximum likelihood fit.
     - FITRESULT corresponds to the absolute path to the output file of the maximum likelihood fit.
     - UNCERTS corresponds to a dictionary of uncert. names mapped to uncert values.
     - METRIC corresponds to the chosen metrix. Available options are 'b', 's+b', 'max'.
     - THRESHOLD corresponds to the threshold below which the uncertainty will be drop from the
       datacard.
     - BLACKLIST corresponds to a list of regular expressions indicating uncertianties, which
       should not be considered for pruning but remain untouched (holy cows)
    Retrun value is a list of names for uncertainties to be excluded from the datacards (i.e.
    pruned) and a list of names of nuisance parameters to be kept and an integer indicating in how
    many cases nuisances have been excluded from pruning due to a missmatch between fit results
    file and list of uncertainties from the input datacards. 
    """
    file= open(FITRESULT,'r')
    confused = 0
    keep = []
    drop = []
    def save(name) :
        """
        Return True if name doews have a correspondence in the list of
        regular expressions in BLACKLIST and False else.
        """
        inList = False
        for unc in BLACKLIST :
            if unc == '' :
                continue
            if re.search(unc, name) :
                inList = True
        return inList
    
    for line in file :
        ## first element is the name of the nuisance parameter
        name=line.split()[0]
        if name == 'name' or name == 'r' :
            continue
        missmatch = False
        if not name in UNCERTS :
            confused += 1
            missmatch = True
            print "Warning: uncertainty:", name,  " found in output file of maximum likelihood fit but not in list of uncertainties as defined in datacards."
        pull_pattern = re.compile('[+-]\d+\.\d+(?=sig)')
        pulls = pull_pattern.findall(line)
        if pulls :
            val = 0.
            ## define metric
            if METRIC == 'b' :
                val = float(pulls[0])
            if METRIC == 's+b' :
                val = float(pulls[1])
            if METRIC == 'max' :
                val = max(abs(float(pulls[0])), float(pulls[1]))
            if not missmatch :
                val*= UNCERTS[name]
            else :
                val = 99999.
            #print name, "->", abs(val)
            if abs(val) < float(THRESHOLD) :
                if save(name) :
                    keep.append(name)
                else :
                    drop.append(name)
            else :
                keep.append(name)
    file.close()
    return (drop, keep, confused)

def list_to_file(LIST, FILE) :
    """
    Write all  elements in a list to a file.
     - LIST corresponds to the list of elements.
     - FILE corresponds to the output file.
    There is no return value 
    """
    file= open(FILE,'w')
    for element in LIST :
        file.write(element+'\n')
    file.close()

def manipulate_datacard(DATACARD, MANIPULATION, EXCLUDE=None) :
    """
    Manipulate DATACARD. Possible manipulations are to uncomment all uncertainties in the datacard or to
    comment those uncertainties which are element of the list EXCLUDE. If the list EXCLUDE is None all
    uncertainties will be commented.
     - DATACARD is the full path to the datacard that is to be manipulated.
     - MANIPULATION indicates the manipulation that should be applied. Possible choices are 'UNCOMMENT'
       and 'COMMENT'.
     - EXCLUDE is a list of uncertainties that are to be commented from DATACARD.
    Return value is the number of uncertaintie that have been manipulated. Note that this function will
    alter DATACARD.
    """
    excl=0
    file = open(DATACARD,'r')
    rnd_name=''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(10))
    output = open('/tmp/{NAME}'.format(NAME=rnd_name), 'w')
    for line in file :
        words = line.split()
        if len(words) > 1 :
            if 'shape' in words[1] or 'lnN' in words[1] :
                name = words[0]
                if MANIPULATION == "COMMENT" :
                    if EXCLUDE != None:
                        if name in EXCLUDE :
                            excl+=1
                            line = '#'+line                    
                        else :
                            excl+=1
                            line = '#'+line
                if MANIPULATION == "UNCOMMENT" :
                    excl+=1
                    line = line.lstrip('#')
        output.write(line)
    file.close()
    output.close()
    os.system("mv /tmp/{NAME} {DATACARD}".format(NAME=rnd_name, DATACARD=DATACARD))
    return excl

### THE MAIN FUNCTION STARTS HERE ###
def main() :
    ## turn options.fit_results into a list
    fit_results = options.fit_results.replace('$PWD', os.getcwd()).split(' ')
    for idx in range(len(fit_results)) : fit_results[idx] = fit_results[idx].rstrip(',')
    ## turn options.blacklist into a list
    blacklist = options.blacklist.split(' ')
    for idx in range(len(blacklist)) : blacklist[idx] = blacklist[idx].rstrip(',')
    ## START
    print "# --------------------------------------------------------------------------------------"
    print "# Pruning uncertainties. "
    print "# --------------------------------------------------------------------------------------"
    print "# You are using the following configuration: "
    print "# --fit-results       :", fit_results
    print "# --metric            :", options.metric
    print "# --mass              :", options.mass
    print "# --threshold         :", options.threshold
    print "# --blacklist         :", options.blacklist
    print "# --comment-nuisances :", options.comment_nuisances
    print "# Check option --help in case of doubt about the meaning of one or more of these confi-"
    print "# guration parameters.                           "
    print "# --------------------------------------------------------------------------------------"
    ## create a combined datacard from input datacards
    rnd_name=''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(10))
    os.system("combineCards.py -S {PATH}/*.txt > /tmp/{NAME}".format(PATH=args[0], NAME=rnd_name))
    ## determine list of all uncertainties from input datacards 
    uncerts = determine_uncerts("/tmp/{NAME}".format(NAME=rnd_name), options.mass)
    ## interface one or more fit_results
    combined_fit_result = combine_fit_results(fit_results, options.metric)
    ## determine list of dropped and kept uncertainties from input datacards
    (dropped, kept, confused) = prune(combined_fit_result, uncerts, options.metric, options.threshold, blacklist)
    ## write dropped and kept uncertainties to file
    list_to_file(kept   , "uncertainty-pruning-keep.txt")
    list_to_file(dropped, "uncertainty-pruning-drop.txt")
    ## comment dropped uncertainties from datacards if configured such
    if options.comment_nuisances :
        num = 0
        for file in glob.glob(args[0]+'/*.txt') : 
            num = manipulate_datacard(file, "COMMENT", dropped)
    ## cleanup of tmp (delete combined card)
    os.system("rm /tmp/{NAME}".format(NAME=rnd_name))
    print "# Excluded", len(dropped), "uncertainties from", len(dropped)+len(kept), ": (", confused, "not pruned due to missmatch of inputs)."
    print "# Check the output files uncertainty-pruning-keep.txt and uncertainty-pruning-drop.txt"
    print "# for the full list of pruned and and kept parameters.                                "
    print "# --------------------------------------------------------------------------------------"
    
main()
exit(0)
