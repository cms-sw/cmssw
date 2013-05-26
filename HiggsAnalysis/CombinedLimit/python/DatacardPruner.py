import os
import re
import glob
import math
import ROOT
import random
import string

class DatacardPruner(object) :
    """
    Description:
    
    This is a class to prune uncertainties of an existing set of datacards. In pruning decision is based on the relative shift
    of the nuisance parameter by the maximum likelihood fit. Nuisance parameters with a shift below a certaint threshold are
    added to a list of parameters to be pruned. The pruning step needs as inputs the directory that contains all datacards,
    which are supposed to be processed and the output(s) of one (or more) maximum likelihood fit(s) that have been processed
    by the script diffNuisances.py to give a result of all pulls in txt format. A list of these fit results should be made
    available to the class via the list fit_results. If this list contains more than one element for correlated uncertainties,
    which appear in more than one file the pruning decision will be based on the maximal shift in the individual files. The
    class optionally takes a list of python style regular expressions for uncertainties that should not be considered for
    pruning (blacklist) and a list of uncertainties to which the pruning should be restricted (whitelist). The pruning takes
    place based on the relative shift of the nuisance parameter in the maximum likelihood fit(s). The metric can be switched
    between the result of the background-only fit ('b'), the signal-plus-background fit ('s+b') or the maximum of the two
    ('max'). The threshold on this shift, below which parameter will be added to a list of nuisance parameters to be pruned,
    is given the parameter threshold. You can optionally choose to have these uncertainties already commented in the tested
    datacards at the same time. The output of the script is a list of pruned and a list of kept nuisance parameters. In the
    current implementation the script is meant to be used for datacards for counting experiments or for shape analyses based
    on histograms.
    """
    def __init__(self, fit_results, metric='max', mass='125', threshold='0.05', blacklist=[], whitelist=[], comment_nuisances=False) :
        ## list of paths to one or more output files of the max-likelihood fit with combine
        self.fit_results = fit_results
        ## metric to be used for the pruning decision. Expected values are 'b', 's+b', 'max'
        self.metric = metric
        ## mass value of the Higgs boson, when considering the s+b hypothesis in the metric 
        self.mass = mass
        ## threshold on the relative shift of the parameter, below which a parameters will be pruned
        self.threshold = threshold
        ## list of python style regex for nuisance parameters that should not be considered for pruning (holy cows)
        self.blacklist = blacklist
        ## list of python style regex for nuisance parameters that should only be considered for pruning
        self.whitelist = whitelist
        ## is true if the the nuisance parameters should be commented from the test datacards right after the pruning decision has been taken  
        self.comment_nuisances = comment_nuisances
        
    def combine_fit_results(self, FITRESULTS) :
        """
        uses: FITRESULTS (list of one or more files containing the pulls of the max-likelihood fit), self.metric
        Create a pseudo file of fit results form a list of fit results based on a subset of datacards. From multiply occuring
        uncertainties larger pulls will replace smaller pulls according to the corresponding metric. The combined pseudo file
        will be written to the tmp directory. The return value will be the full path to the combined pseudo file. 
        """
        output = {}
        headline = ''
        pull_pattern = re.compile('[+-]\d+\.\d+(?=sig)')
        for fit_result in self.fit_results :
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
                        if self.metric == 'b' :
                            value_old = float(pulls_old[0])
                            value_new = float(pulls_new[0])
                        if self.metric == 's+b' :
                            value_old = float(pulls_old[1])
                            value_new = float(pulls_new[1])
                        if self.metric == 'max' :
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

    def determine_shapes(self, DATACARD) :
        """
        uses: DATACARD (absolute path to the datacard), self.mass
        Determine all shape uncertainties from a given datacard. For all shape uncertainties the largest relative
        uncertainty over all bins is determined and added to a dictionary. Return value is the dictionary mapping
        the name of the uncertainty to the maximal relative uncertainty.
        """
        def valid(u_value, s_value, exceptions) :
            """
            Check for a weak equality between u_value and s_value. The argument exceptions corresponds to exceptional
            cases for which '*' is not true. This function is used for proc and bin.
            """
            if u_value in exceptions :
                return u_value == s_value
            else :
                return u_value == s_value or s_value == '*'
        
        ## determine list of bin names, list of process names and list of input
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
                if not words[1].lstrip('-').isdigit() :
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
                        value = root_file.Get(s_shape.replace('$CHANNEL', u_bin).replace('$PROCESS', u_proc).replace('$MASS', self.mass))
                        upper = root_file.Get(s_syst.replace('$CHANNEL', u_bin).replace('$PROCESS', u_proc).replace('$MASS', self.mass).replace('$SYSTEMATIC', u_unc)+'Up')
                        lower = root_file.Get(s_syst.replace('$CHANNEL', u_bin).replace('$PROCESS', u_proc).replace('$MASS', self.mass).replace('$SYSTEMATIC', u_unc)+'Down')
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

    def determine_lnNs(self, DATACARD):
        """
        uses: DATACARD (absolute path to the datacard)
        Determine all lnN uncertainties from a given datacard. For correlated uncertainties the largest relative
        uncertainty over all occurences is determined and added to a dictionary. eturn value is the dictionary
        mapping the name of the uncertainty to the maximal relative uncertainty.
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

    def determine_uncerts(self, DATACARD) :
        """
        uses: DATACARD (absolute path to the datacard), self.mass
        Determine full list of uncertainties as a dictionary mapping uncertainty name to the maximal relative uncertainty.
        Uncertainties of type shape and lnN are supported. Return value is a dictionary mapping the name of the
        uncertainty to the maximal relative uncertainty.    
        """
        ## add lnN uncertainties to dictionary
        uncerts = self.determine_lnNs(DATACARD)
        ## add shape uncertainties to dictionary
        uncerts.update(self.determine_shapes(DATACARD))
        return uncerts

    def in_list(self, NAME, LIST) :
        """
        uses: NAME (name of an uncertainty), LIST (a list of regex of names of uncertainties)
        Return True if name does have a correspondence in the list of regular expressions of LIST and False else.
        Make sure that LIST does not contain empty strings. 
        """
        inList = False
        for unc in LIST :
            if re.search(unc, NAME) :
                inList = True
        return inList

    def prune(self, UNCERTS) :
        """
        uses: UNCERTS (dictionary of uncertainty names mapped to uncertainty values, self.fit_results, self.metric,
        self.theshold, self.blacklist, self.whitelist
        Take the pruning decision by relative shift of the uncertainty by the maximum likelihood fit. Retrun value is a
        list of names for uncertainties to be pruned (=excluded) from the datacards and a list of names of nuisance
        parameters to be kept and an integer indicating in how many cases a nuisances parameter listed in FITRESULTS did
        NOT have any correspondence in the list of keys of UNCERTS.
        """
        keep = []
        drop = []
        confused = 0
        file_name = self.combine_fit_results(self.fit_results)
        file = open(file_name,'r')
        pull_pattern = re.compile('[+-]\d+\.\d+(?=sig)')
        for line in file :
            ## first element is the name of the nuisance parameter
            name=line.split()[0]
            if name == 'name' or name == 'r' :
                continue
            missmatch = False
            if not name in UNCERTS :
                confused += 1
                missmatch = True
                print "Warning: uncertainty:", name,  " found in output file of maximum likelihood fit but NOT in list of uncertainties as defined by datacards."
            pulls = pull_pattern.findall(line)
            if pulls :
                val = 0.
                ## determine value of pull according to metric
                if self.metric == 'b' :
                    val = float(pulls[0])
                if self.metric == 's+b' :
                    val = float(pulls[1])
                if self.metric == 'max' :
                    val = max(abs(float(pulls[0])), float(pulls[1]))
                if not missmatch :
                    val*= UNCERTS[name]
                else :
                    val = 99999.
                #print name, "->", abs(val)
                if abs(val) < float(self.threshold) :
                    if not len(self.whitelist) == 0 :
                        if not self.in_list(name, self.whitelist) :
                            #print "keep --> not in whilelist:", name, self.in_list(name, self.whitelist)
                            keep.append(name)
                        else :
                            if not self.in_list(name, self.blacklist) :
                                #print "drop -->     in whilelist:", name, self.in_list(name, self.whitelist)
                                drop.append(name)
                            else :
                                #print "keep -->     in blacklist:", name
                                keep.append(name)
                    else :
                        if not self.in_list(name, self.blacklist) :
                            #print "drop --> not in blacklist:", name
                            drop.append(name)
                        else :
                            #print "keep -->     in blacklist:", name
                            keep.append(name)
                else :
                    keep.append(name)
        file.close()
        #print "wrote combined cards to: {NAME}".format(NAME=file_name)
        os.system("rm {NAME}".format(NAME=file_name))
        return (drop, keep, confused)

    def list_to_file(self, LIST, FILE) :
        """
        uses: LIST (list of elements), FILE (path to an output file in txt format)
        Write all elements in a LIST to FILE. There is no return value. 
        """
        file= open(FILE,'w')
        for element in LIST :
            file.write(element+'\n')
        file.close()
        return
        
    def manipulate_datacard(self, DATACARD, MANIPULATION, EXCLUDE=None) :
        """
        uses: DATACARD (path to the input datacard), MANIPULATION (a manipulation string), EXCLUDE (list of nuicance
        parameters to be commented in the datacard).
        Manipulate DATACARD. Possible manipulations are to uncomment all uncertainties in the datacard or to comment
        those uncertainties, which are element of EXCLUDE. Possible manipulations are passed by the string MANIPULATION.
        This string can take the values COMMENT or UNCOMMENT. If the value of MANIPULATION is COMMENT and EXCLUDE is
        None, all uncertainties in the datacards will be commented. Return value is the number of uncertainties that have
        been manipulated. Note that this function will alter DATACARD.
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
                    elif MANIPULATION == "UNCOMMENT" :
                        excl+=1
                        line = line.lstrip('#')
                    else :
                        print "Warning: MANIPULATION:", MANIPULATION, "unknown. Possible values are: COMMENT, UNCOMMENT."
            output.write(line)
        file.close()
        output.close()
        os.system("mv /tmp/{NAME} {DATACARD}".format(NAME=rnd_name, DATACARD=DATACARD))
        return excl
