from ROOT import TFile, TH1D
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, default="debug.root", help="Input file name (include '.root' suffix). Full or relative directory path allowed.")
parser.add_argument("--maxEvents", type=int, default=20, help="Define maximum number of events to run on.")
parser.add_argument("--dontRunTCs", action="store_true", default=False, help="Don't check for duplicates in the TC collection.")
parser.add_argument("--runExtraObjects", action="store_true", default=False, help="Also check for duplicates in the full pT5, pT3 and T5 collections.")
parser.add_argument("--debugLevel", type=int, default=0, choices=[0,1,2], help="0: No debug output, 1: Event per event output, 2: Object level output.")
args = parser.parse_args()


infile = TFile(args.infile,"read")
intree = infile.Get("tree")


def simTrkInfo(event,dict_):
  dict_["N"] = dict_["N"] + len(event.sim_pt)
  for simTrk in range(len(event.sim_pt)):
    isGood = False
    if event.sim_isGood[simTrk]:
      isGood = True
      dict_["N_good"] = dict_["N_good"] + 1
    if event.sim_TC_matched[simTrk] > 0:
      dict_["N_matched"] = dict_["N_matched"] + 1
      if isGood: dict_["N_goodMatched"] = dict_["N_goodMatched"] + 1
      if event.sim_TC_matched[simTrk] == 1:
        dict_["N_singleMatched"] = dict_["N_singleMatched"] + 1
        if isGood: dict_["N_goodSingleMatched"] = dict_["N_goodSingleMatched"] + 1
      if event.sim_TC_matched[simTrk] > 1:
        dict_["N_dup"] = dict_["N_dup"] + 1
        if isGood: dict_["N_goodDup"] = dict_["N_goodDup"] + 1
      for tc in range(len(event.tc_matched_simIdx)):
        for simIdxOther in range(len(event.tc_matched_simIdx[tc])):
          if event.tc_matched_simIdx[tc][simIdxOther] == simTrk:
            if event.tc_type[tc] == 7:
              dict_["N_matchedWpT5"] = dict_["N_matchedWpT5"] + 1
              if isGood: dict_["N_goodMatchedWpT5"] = dict_["N_goodMatchedWpT5"] + 1
            if event.tc_type[tc] == 5:
              dict_["N_matchedWpT3"] = dict_["N_matchedWpT3"] + 1
              if isGood: dict_["N_goodMatchedWpT3"] = dict_["N_goodMatchedWpT3"] + 1
            if event.tc_type[tc] == 4:
              dict_["N_matchedWT5"] = dict_["N_matchedWT5"] + 1
              if isGood: dict_["N_goodMatchedWT5"] = dict_["N_goodMatchedWT5"] + 1
            if event.tc_type[tc] == 8:
              dict_["N_matchedWpLS"] = dict_["N_matchedWpLS"] + 1
              if isGood: dict_["N_goodMatchedWpLS"] = dict_["N_goodMatchedWpLS"] + 1

  return dict_


def dupOfTCObj(tc,objType,event,dict_,debug=False):
  isDuplicate = False
  isDuplicateWpT5 = False
  isDuplicateWpT3 = False
  isDuplicateWT5 = False
  isDuplicateWpLS = False

  dict_["N_"+objType] = dict_["N_"+objType] + 1

  if event.tc_isFake[tc] == 1: dict_["N_"+objType+"fakes"] = dict_["N_"+objType+"fakes"] + 1
  for simIdx in range(len(event.tc_matched_simIdx[tc])):
    for tcOther in range(len(event.tc_pt)):
      if tc==tcOther: continue
      for simIdxOther in range(len(event.tc_matched_simIdx[tcOther])):
        if event.tc_matched_simIdx[tc][simIdx] == event.tc_matched_simIdx[tcOther][simIdxOther]:
          isDuplicate = True
          if debug:
            print ""
            print "TC_"+objType+"[%d] pt = %.2f, eta = %.2f, phi = %.2f" %( tc, event.tc_pt[tc], event.tc_eta[tc], event.tc_phi[tc] )
            print "Matched simTrk pt = %.2f" %event.sim_pt[event.tc_matched_simIdx[tc][simIdx]]
            print "Duplicate of type %d:" %event.tc_type[tcOther]
            print "\tTC_[%d] pt = %.2f, eta = %.2f, phi = %.2f" %( tcOther, event.tc_pt[tcOther], event.tc_eta[tcOther], event.tc_phi[tcOther] )
          if event.tc_type[tcOther] == 7: isDuplicateWpT5 = True
          if event.tc_type[tcOther] == 5: isDuplicateWpT3 = True
          if event.tc_type[tcOther] == 4: isDuplicateWT5 = True
          if event.tc_type[tcOther] == 8: isDuplicateWpLS = True

  if isDuplicate: dict_["Ndup_"+objType+"Total"] = dict_["Ndup_"+objType+"Total"] + 1
  if isDuplicateWpT5: dict_["Ndup_"+objType+"WpT5"] = dict_["Ndup_"+objType+"WpT5"] + 1
  if isDuplicateWpT3: dict_["Ndup_"+objType+"WpT3"] = dict_["Ndup_"+objType+"WpT3"] + 1
  if isDuplicateWT5: dict_["Ndup_"+objType+"WT5"] = dict_["Ndup_"+objType+"WT5"] + 1
  if isDuplicateWpLS: dict_["Ndup_"+objType+"WpLS"] = dict_["Ndup_"+objType+"WpLS"] + 1

  return dict_


def dupOfTC(event,dict_,debug=False):
  for tc in range(len(event.tc_pt)):
    if event.tc_type[tc] == 7: dict_ = dupOfTCObj(tc,"pT5",event,dict_,debug=debug)
    if event.tc_type[tc] == 5: dict_ = dupOfTCObj(tc,"pT3",event,dict_,debug=debug)
    if event.tc_type[tc] == 4: dict_ = dupOfTCObj(tc,"T5",event,dict_,debug=debug)
    if event.tc_type[tc] == 8: dict_ = dupOfTCObj(tc,"pLS",event,dict_,debug=debug)

  return dict_


def dupOfpT5(event,dict_,debug=False):
  if debug: print "Number of pT5: ",len(event.pT5_pt)
  dict_["N"] = dict_["N"] + len(event.pT5_pt)
  for pT5 in range(len(event.pT5_pt)):
    isDuplicateWpT5 = False
    isDuplicateWpT3 = False
    isDuplicateWT5 = False

    if event.pT5_isFake[pT5] == 1: dict_["Nfakes"] = dict_["Nfakes"] + 1
    if event.pT5_isDuplicate[pT5] == 1: isDuplicateWpT5 = True

    for j in range(len(event.pT5_matched_simIdx[pT5])):
      if debug:
        if event.pT5_isDuplicate[pT5]:
          print "pT5[",pT5,"]"
          print "simTrk_pt = ",event.sim_pt[event.pT5_matched_simIdx[pT5][j]]

      if event.sim_pT3_matched[event.pT5_matched_simIdx[pT5][j]] > 0:
        isDuplicateWpT3 = True
        if debug:
          print ""
          print "pT5_pt = ",event.pT5_pt[pT5]
          print "simTrk_pt = ",event.sim_pt[event.pT5_matched_simIdx[pT5][j]]
          for pT3 in range(len(event.pT3_pt)):
            for k in range(len(event.pT3_matched_simIdx[pT3])):
              if event.pT5_matched_simIdx[pT5][j] == event.pT3_matched_simIdx[pT3][k]:
                print ""
                print "pT3_pt = ",event.pT3_pt[pT3]
                print "simTrk_pt = ",event.sim_pt[event.pT3_matched_simIdx[pT3][k]]
                
      if event.sim_T5_matched[event.pT5_matched_simIdx[pT5][j]] > 0:
        isDuplicateWT5 = True
        if debug:
          print ""
          print "pT5_pt = ",event.pT5_pt[pT5]
          print "simTrk_pt = ",event.sim_pt[event.pT5_matched_simIdx[pT5][j]]
          for T5 in range(len(event.t5_pt)):
            for k in range(len(event.t5_matched_simIdx[T5])):
              if event.pT5_matched_simIdx[pT5][j] == event.t5_matched_simIdx[T5][k]:
                print ""
                print "T5_pt = ",event.t5_pt[T5]
                print "simTrk_pt = ",event.sim_pt[event.t5_matched_simIdx[T5][k]]

    if isDuplicateWpT5:
      if isDuplicateWpT3:
        if isDuplicateWT5:
          dict_["Ndup_Wall"] = dict_["Ndup_Wall"] + 1
        else:
          dict_["Ndup_WpT5ApT3"] = dict_["Ndup_WpT5ApT3"] + 1
      else:
        if isDuplicateWT5:
          dict_["Ndup_WpT5AT5"] = dict_["Ndup_WpT5AT5"] + 1
        else:
          dict_["Ndup_WpT5"] = dict_["Ndup_WpT5"] + 1
    else:
      if isDuplicateWpT3:
        if isDuplicateWT5:
          dict_["Ndup_WpT3AT5"] = dict_["Ndup_WpT3AT5"] + 1
        else:
          dict_["Ndup_WpT3"] = dict_["Ndup_WpT3"] + 1
      else:
        if isDuplicateWT5:
          dict_["Ndup_WT5"] = dict_["Ndup_WT5"] + 1

  return dict_


def dupOfpT3(event,dict_,debug=False):
  if debug: print "Number of pT3: ",len(event.pT3_pt)
  dict_["N"] = dict_["N"] + len(event.pT3_pt)
  for pT3 in range(len(event.pT3_pt)):
    isDuplicateWpT5 = False
    isDuplicateWpT3 = False
    isDuplicateWT5 = False

    if event.pT3_isFake[pT3] == 1: dict_["Nfakes"] = dict_["Nfakes"] + 1
    if event.pT3_isDuplicate[pT3] == 1: isDuplicateWpT3 = True

    for j in range(len(event.pT3_matched_simIdx[pT3])):
      if debug:
        if event.pT3_isDuplicate[pT3]:
          print "pT3[",pT3,"]"
          print "simTrk_pt = ",event.sim_pt[event.pT3_matched_simIdx[pT3][j]]

      if event.sim_pT5_matched[event.pT3_matched_simIdx[pT3][j]] > 0:
        isDuplicateWpT5 = True
        if debug:
          print ""
          print "pT3_pt = ",event.pT3_pt[pT3]
          print "simTrk_pt = ",event.sim_pt[event.pT3_matched_simIdx[pT3][j]]
          for pT5 in range(len(event.pT5_pt)):
            for k in range(len(event.pT5_matched_simIdx[pT5])):
              if event.pT3_matched_simIdx[pT3][j] == event.pT5_matched_simIdx[pT5][k]:
                print ""
                print "pT5_pt = ",event.pT5_pt[pT5]
                print "simTrk_pt = ",event.sim_pt[event.pT5_matched_simIdx[pT5][k]]
                
      if event.sim_T5_matched[event.pT3_matched_simIdx[pT3][j]] > 0:
        isDuplicateWT5 = True
        if debug:
          print ""
          print "pT3_pt = ",event.pT3_pt[pT3]
          print "simTrk_pt = ",event.sim_pt[event.pT3_matched_simIdx[pT3][j]]
          for T5 in range(len(event.t5_pt)):
            for k in range(len(event.t5_matched_simIdx[T5])):
              if event.pT3_matched_simIdx[pT3][j] == event.t5_matched_simIdx[T5][k]:
                print ""
                print "T5_pt = ",event.t5_pt[T5]
                print "simTrk_pt = ",event.sim_pt[event.t5_matched_simIdx[T5][k]]

    if isDuplicateWpT5:
      if isDuplicateWpT3:
        if isDuplicateWT5:
          dict_["Ndup_Wall"] = dict_["Ndup_Wall"] + 1
        else:
          dict_["Ndup_WpT5ApT3"] = dict_["Ndup_WpT5ApT3"] + 1
      else:
        if isDuplicateWT5:
          dict_["Ndup_WpT5AT5"] = dict_["Ndup_WpT5AT5"] + 1
        else:
          dict_["Ndup_WpT5"] = dict_["Ndup_WpT5"] + 1
    else:
      if isDuplicateWpT3:
        if isDuplicateWT5:
          dict_["Ndup_WpT3AT5"] = dict_["Ndup_WpT3AT5"] + 1
        else:
          dict_["Ndup_WpT3"] = dict_["Ndup_WpT3"] + 1
      else:
        if isDuplicateWT5:
          dict_["Ndup_WT5"] = dict_["Ndup_WT5"] + 1

  return dict_


def dupOfT5(event,dict_,debug=False):
  if debug: print "Number of T5: ",len(event.t5_pt)
  dict_["N"] = dict_["N"] + len(event.t5_pt)
  for t5 in range(len(event.t5_pt)):
    isDuplicateWpT5 = False
    isDuplicateWpT3 = False
    isDuplicateWT5 = False

    if event.t5_isFake[t5] == 1: dict_["Nfakes"] = dict_["Nfakes"] + 1
    if event.t5_isDuplicate[t5] == 1: isDuplicateWT5 = True

    for j in range(len(event.t5_matched_simIdx[t5])):
      if debug:
        if event.t5_isDuplicate[t5]:
          print "T5[",t5,"]"
          print "simTrk_pt = ",event.sim_pt[event.t5_matched_simIdx[t5][j]]

      if event.sim_pT5_matched[event.t5_matched_simIdx[t5][j]] > 0:
        isDuplicateWpT5 = True
        if debug:
          print ""
          print "T5_pt = ",event.t5_pt[t5]
          print "simTrk_pt = ",event.sim_pt[event.t5_matched_simIdx[t5][j]]
          for pT5 in range(len(event.pT5_pt)):
            for k in range(len(event.pT5_matched_simIdx[pT5])):
              if event.t5_matched_simIdx[t5][j] == event.pT5_matched_simIdx[pT5][k]:
                print ""
                print "pT5_pt = ",event.pT5_pt[pT5]
                print "simTrk_pt = ",event.sim_pt[event.pT5_matched_simIdx[pT5][k]]
                
      if event.sim_pT3_matched[event.t5_matched_simIdx[t5][j]] > 0:
        isDuplicateWpT3 = True
        if debug:
          print ""
          print "T5_pt = ",event.t5_pt[t5]
          print "simTrk_pt = ",event.sim_pt[event.t5_matched_simIdx[t5][j]]
          for pT3 in range(len(event.pT3_pt)):
            for k in range(len(event.pT3_matched_simIdx[pT3])):
              if event.t5_matched_simIdx[t5][j] == event.pT3_matched_simIdx[pT3][k]:
                print ""
                print "pT3_pt = ",event.pT3_pt[pT3]
                print "simTrk_pt = ",event.sim_pt[event.pT3_matched_simIdx[pT3][k]]

    if isDuplicateWpT5:
      if isDuplicateWpT3:
        if isDuplicateWT5:
          dict_["Ndup_Wall"] = dict_["Ndup_Wall"] + 1
        else:
          dict_["Ndup_WpT5ApT3"] = dict_["Ndup_WpT5ApT3"] + 1
      else:
        if isDuplicateWT5:
          dict_["Ndup_WpT5AT5"] = dict_["Ndup_WpT5AT5"] + 1
        else:
          dict_["Ndup_WpT5"] = dict_["Ndup_WpT5"] + 1
    else:
      if isDuplicateWpT3:
        if isDuplicateWT5:
          dict_["Ndup_WpT3AT5"] = dict_["Ndup_WpT3AT5"] + 1
        else:
          dict_["Ndup_WpT3"] = dict_["Ndup_WpT3"] + 1
      else:
        if isDuplicateWT5:
          dict_["Ndup_WT5"] = dict_["Ndup_WT5"] + 1

  return dict_


def printSimComp(dict_):
  if dict_["N"] == 0:
    print "No sim  object found!"
    return

  print ""
  print "Total sim multiplicity = %d" %dict_["N"]
  print "Matched sim = %d (%.2f%%)" %( dict_["N_matched"], float(dict_["N_matched"])/float(dict_["N"])*100 )
  print "Single matched sim = %d (%.2f%%)" %( dict_["N_singleMatched"], float(dict_["N_singleMatched"])/float(dict_["N"])*100 )
  print "Duplicate sim = %d (%.2f%%)" %( dict_["N_dup"], float(dict_["N_dup"])/float(dict_["N"])*100 )
  print "Matched with pT5 = %d (%.2f%%)" %( dict_["N_matchedWpT5"], float(dict_["N_matchedWpT5"])/float(dict_["N"])*100 )
  print "Matched with pT3 = %d (%.2f%%)" %( dict_["N_matchedWpT3"], float(dict_["N_matchedWpT3"])/float(dict_["N"])*100 )
  print "Matched with T5 = %d (%.2f%%)" %( dict_["N_matchedWT5"], float(dict_["N_matchedWT5"])/float(dict_["N"])*100 )
  print "Matched with pLS = %d (%.2f%%)" %( dict_["N_matchedWpLS"], float(dict_["N_matchedWpLS"])/float(dict_["N"])*100 )
  print ""
  print "Good sim = %d (%.2f%%)" %( dict_["N_good"], float(dict_["N_good"])/float(dict_["N"])*100 )
  print "Matched good sim = %d (%.2f%%)" %( dict_["N_goodMatched"], float(dict_["N_goodMatched"])/float(dict_["N_good"])*100 )
  print "Single matched good sim = %d (%.2f%%)" %( dict_["N_goodSingleMatched"], float(dict_["N_goodSingleMatched"])/float(dict_["N_good"])*100 )
  print "Duplicate good sim = %d (%.2f%%)" %( dict_["N_goodDup"], float(dict_["N_goodDup"])/float(dict_["N_good"])*100 )
  print "Good sim matched with pT5 = %d (%.2f%%)" %( dict_["N_goodMatchedWpT5"], float(dict_["N_goodMatchedWpT5"])/float(dict_["N_good"])*100 )
  print "Good sim matched with pT3 = %d (%.2f%%)" %( dict_["N_goodMatchedWpT3"], float(dict_["N_goodMatchedWpT3"])/float(dict_["N_good"])*100 )
  print "Good sim matched with T5 = %d (%.2f%%)" %( dict_["N_goodMatchedWT5"], float(dict_["N_goodMatchedWT5"])/float(dict_["N_good"])*100 )
  print "Good sim matched with pLS = %d (%.2f%%)" %( dict_["N_goodMatchedWpLS"], float(dict_["N_goodMatchedWpLS"])/float(dict_["N_good"])*100 )
  print ""
  return


def printTCComp(objType,dict_):
  if dict_["N_"+objType] == 0:
    print "No "+objType+" object found in TC collection!"
    return
  dict_["N_"+objType+"singleMatched"] = dict_["N_"+objType]-dict_["N_"+objType+"fakes"]-dict_["Ndup_"+objType+"Total"]

  print ""
  print "Total "+objType+" multiplicity in TC collection = %d" %dict_["N_"+objType]
  print objType+" Fakes = %d (%.2f%%)" %( dict_["N_"+objType+"fakes"], float(dict_["N_"+objType+"fakes"])/float(dict_["N_"+objType])*100 )
  print objType+" Duplicates with pT5 = %d (%.2f%%)" %( dict_["Ndup_"+objType+"WpT5"], float(dict_["Ndup_"+objType+"WpT5"])/float(dict_["N_"+objType])*100 )
  print objType+" Duplicates with pT3 = %d (%.2f%%)" %( dict_["Ndup_"+objType+"WpT3"], float(dict_["Ndup_"+objType+"WpT3"])/float(dict_["N_"+objType])*100 )
  print objType+" Duplicates with T5 = %d (%.2f%%)" %( dict_["Ndup_"+objType+"WT5"], float(dict_["Ndup_"+objType+"WT5"])/float(dict_["N_"+objType])*100 )
  print objType+" Duplicates with pLS = %d (%.2f%%)" %( dict_["Ndup_"+objType+"WpLS"], float(dict_["Ndup_"+objType+"WpLS"])/float(dict_["N_"+objType])*100 )
  print objType+" Total duplicates = %d (%.2f%%)" %( dict_["Ndup_"+objType+"Total"], float(dict_["Ndup_"+objType+"Total"])/float(dict_["N_"+objType])*100 )
  print objType+" Single matched non-fakes = %d (%.2f%%)" %( dict_["N_"+objType+"singleMatched"], float(dict_["N_"+objType+"singleMatched"])/float(dict_["N_"+objType])*100 )
  print ""
  return


def printObjComp(objType,dict_):
  if dict_["N"] == 0:
    print "No "+objType+" object found!"
    return
  dict_["Ndup_Total"] = dict_["Ndup_WpT5"]+dict_["Ndup_WpT3"]+dict_["Ndup_WT5"]+dict_["Ndup_WpT5ApT3"]+dict_["Ndup_WpT5AT5"]+dict_["Ndup_WpT3AT5"]+dict_["Ndup_Wall"]
  dict_["NsingleMatched"] = dict_["N"]-dict_["Nfakes"]-dict_["Ndup_Total"]

  print ""
  print "Total "+objType+" multiplicity = %d" %dict_["N"]
  print objType+" Fakes = %d (%.2f%%)" %( dict_["Nfakes"], float(dict_["Nfakes"])/float(dict_["N"])*100 )
  print objType+" Duplicates with pT5 only = %d (%.2f%%)" %( dict_["Ndup_WpT5"], float(dict_["Ndup_WpT5"])/float(dict_["N"])*100 )
  print objType+" Duplicates with pT3 only = %d (%.2f%%)" %( dict_["Ndup_WpT3"], float(dict_["Ndup_WpT3"])/float(dict_["N"])*100 )
  print objType+" Duplicates with T5 only = %d (%.2f%%)" %( dict_["Ndup_WT5"], float(dict_["Ndup_WT5"])/float(dict_["N"])*100 )
  print objType+" Duplicates with pT5 and pT3 only = %d (%.2f%%)" %( dict_["Ndup_WpT5ApT3"], float(dict_["Ndup_WpT5ApT3"])/float(dict_["N"])*100 )
  print objType+" Duplicates with pT5 and T5 only = %d (%.2f%%)" %( dict_["Ndup_WpT5AT5"], float(dict_["Ndup_WpT5AT5"])/float(dict_["N"])*100 )
  print objType+" Duplicates with pT3 and T5 only = %d (%.2f%%)" %( dict_["Ndup_WpT3AT5"], float(dict_["Ndup_WpT3AT5"])/float(dict_["N"])*100 )
  print objType+" Duplicates with all = %d (%.2f%%)" %( dict_["Ndup_Wall"], float(dict_["Ndup_Wall"])/float(dict_["N"])*100 )
  print objType+" Total duplicates = %d (%.2f%%)" %( dict_["Ndup_Total"], float(dict_["Ndup_Total"])/float(dict_["N"])*100 )
  print objType+" Single matched non-fakes = %d (%.2f%%)" %( dict_["NsingleMatched"], float(dict_["NsingleMatched"])/float(dict_["N"])*100 )
  print ""
  return


dict_sim = { "N": 0, "N_matched": 0, "N_singleMatched": 0, "N_dup": 0, "N_matchedWpT5": 0, "N_matchedWpT3": 0, "N_matchedWT5": 0, "N_matchedWpLS": 0,\
    "N_good": 0, "N_goodMatched": 0, "N_goodSingleMatched": 0, "N_goodDup": 0, "N_goodMatchedWpT5": 0, "N_goodMatchedWpT3": 0, "N_goodMatchedWT5": 0, "N_goodMatchedWpLS": 0 }
dict_TCTot = { "N_": 0, "N_fakes": 0, "Ndup_WpT5": 0, "Ndup_WpT3": 0, "Ndup_WT5": 0, "Ndup_WpLS": 0, "Ndup_Total": 0, "N_singleMatched": 0 }
dict_TC = { "N_pT5": 0, "N_pT5fakes": 0, "Ndup_pT5WpT5": 0, "Ndup_pT5WpT3": 0, "Ndup_pT5WT5": 0, "Ndup_pT5WpLS": 0, "Ndup_pT5Total": 0, "N_pT5singleMatched": 0,\
    "N_pT3": 0, "N_pT3fakes": 0, "Ndup_pT3WpT5": 0, "Ndup_pT3WpT3": 0, "Ndup_pT3WT5": 0, "Ndup_pT3WpLS": 0, "Ndup_pT3Total": 0, "N_pT3singleMatched": 0,\
    "N_T5": 0, "N_T5fakes": 0, "Ndup_T5WpT5": 0, "Ndup_T5WpT3": 0, "Ndup_T5WT5": 0, "Ndup_T5WpLS": 0, "Ndup_T5Total": 0, "N_T5singleMatched": 0,\
    "N_pLS": 0, "N_pLSfakes": 0, "Ndup_pLSWpT5": 0, "Ndup_pLSWpT3": 0, "Ndup_pLSWT5": 0, "Ndup_pLSWpLS": 0, "Ndup_pLSTotal": 0, "N_pLSsingleMatched": 0 }
dict_pT5 = { "N": 0, "Nfakes": 0, "Ndup_WpT5": 0, "Ndup_WpT3": 0, "Ndup_WT5": 0, "Ndup_WpT5ApT3": 0, "Ndup_WpT5AT5": 0, "Ndup_WpT3AT5": 0, "Ndup_Wall": 0, "Ndup_Total": 0, "NsingleMatched": 0 }
dict_pT3 = { "N": 0, "Nfakes": 0, "Ndup_WpT5": 0, "Ndup_WpT3": 0, "Ndup_WT5": 0, "Ndup_WpT5ApT3": 0, "Ndup_WpT5AT5": 0, "Ndup_WpT3AT5": 0, "Ndup_Wall": 0, "Ndup_Total": 0, "NsingleMatched": 0 }
dict_T5 = { "N": 0, "Nfakes": 0, "Ndup_WpT5": 0, "Ndup_WpT3": 0, "Ndup_WT5": 0, "Ndup_WpT5ApT3": 0, "Ndup_WpT5AT5": 0, "Ndup_WpT3AT5": 0, "Ndup_Wall": 0, "Ndup_Total": 0, "NsingleMatched": 0 }

debug = False
if args.debugLevel==2: debug=True # Object level debugging

for i,event in enumerate(intree):
  if i==args.maxEvents: break
  print "Event : %d" %i

  dict_sim = simTrkInfo(event,dict_sim)
  if not args.dontRunTCs: dict_TC = dupOfTC(event,dict_TC,debug=debug)
  if args.runExtraObjects:
    dict_pT5 = dupOfpT5(event,dict_pT5,debug=debug)
    dict_pT3 = dupOfpT3(event,dict_pT3,debug=debug)
    dict_T5 = dupOfT5(event,dict_T5,debug=debug)

  if args.debugLevel>0: # Event per event debugging
    printSimComp(dict_sim)
    printTCComp("pT5",dict_TC)
    printTCComp("pT3",dict_TC)
    printTCComp("T5",dict_TC)
    printTCComp("pLS",dict_TC)
    if args.runExtraObjects:
      printObjComp("pT5",dict_pT5)
      printObjComp("pT3",dict_pT3)
      printObjComp("T5",dict_T5)
    print ""
    if i==2: break

printSimComp(dict_sim)
if not args.dontRunTCs:
  dict_TCTot["N_"] = dict_TC["N_pT5"] + dict_TC["N_pT3"] + dict_TC["N_T5"] + dict_TC["N_pLS"]
  dict_TCTot["N_fakes"] = dict_TC["N_pT5fakes"] + dict_TC["N_pT3fakes"] + dict_TC["N_T5fakes"] + dict_TC["N_pLSfakes"]
  dict_TCTot["Ndup_WpT5"] = dict_TC["Ndup_pT5WpT5"] + dict_TC["Ndup_pT3WpT5"] + dict_TC["Ndup_T5WpT5"] + dict_TC["Ndup_pLSWpT5"]
  dict_TCTot["Ndup_WpT3"] = dict_TC["Ndup_pT5WpT3"] + dict_TC["Ndup_pT3WpT3"] + dict_TC["Ndup_T5WpT3"] + dict_TC["Ndup_pLSWpT3"]
  dict_TCTot["Ndup_WT5"] = dict_TC["Ndup_pT5WT5"] + dict_TC["Ndup_pT3WT5"] + dict_TC["Ndup_T5WT5"] + dict_TC["Ndup_pLSWT5"]
  dict_TCTot["Ndup_WpLS"] = dict_TC["Ndup_pT5WpLS"] + dict_TC["Ndup_pT3WpLS"] + dict_TC["Ndup_T5WpLS"] + dict_TC["Ndup_pLSWpLS"]
  dict_TCTot["Ndup_Total"] = dict_TC["Ndup_pT5Total"] + dict_TC["Ndup_pT3Total"] + dict_TC["Ndup_T5Total"] + dict_TC["Ndup_pLSTotal"]
  dict_TCTot["N_singleMatched"] = dict_TC["N_pT5singleMatched"] + dict_TC["N_pT3singleMatched"] + dict_TC["N_T5singleMatched"] + dict_TC["N_pLSsingleMatched"]

  printTCComp("",dict_TCTot)
  printTCComp("pT5",dict_TC)
  printTCComp("pT3",dict_TC)
  printTCComp("T5",dict_TC)
  printTCComp("pLS",dict_TC)
if args.runExtraObjects:
  printObjComp("pT5",dict_pT5)
  printObjComp("pT3",dict_pT3)
  printObjComp("T5",dict_T5)
