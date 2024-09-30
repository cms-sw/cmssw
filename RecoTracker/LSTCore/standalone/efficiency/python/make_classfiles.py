#!/usr/bin/env python

import ROOT as r
import argparse
import sys
import os

def classname_to_type(cname): return "const " + cname.strip()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="comma separated list of name(s) of file(s) to make classfile on")
    parser.add_argument("-t", "--tree", help="treename (default: Events)", default="Events")
    parser.add_argument("-n", "--namespace", help="namespace (default: tas)", default="tas")
    parser.add_argument("-o", "--objectname", help="objectname (default: cms3)", default="cms3")
    parser.add_argument("-c", "--classname", help="classname (default: CMS3)", default="CMS3")
    parser.add_argument("-l", "--looper", help="make a looper as well", default=False, action="store_true")
    args = parser.parse_args()

    fnames_in = args.filename.split(',')
    treename = args.tree
    classname = args.classname
    objectname = args.objectname
    namespace = args.namespace
    make_looper = args.looper

    ##
    ## create a unique list of branches
    ##
    names = set()
    aliases = r.TList()
    branches = r.TList()
    fs = []
    trees = []

    haveHLTInfo = False
    haveHLT8E29Info = False    
    haveL1Info = False
    haveTauIDInfo = False
    haveBtagInfo = False    
    
    for name in fnames_in:
        f = r.TFile(name)
        tree = f.Get(treename)
        fs.append(f)
        trees.append(tree)
        temp_aliases = trees[-1].GetListOfAliases()
        temp_branches = trees[-1].GetListOfBranches()
        
        if temp_aliases:
            for ala in temp_aliases:
                alias = ala.GetName()
                if alias not in names:
                    names.add(alias)
                    aliases.Add(ala)
                    branches.Add(trees[-1].GetBranch(trees[-1].GetAlias(alias)))
                if "hlt_trigNames" in alias: haveHLTInfo = True
                if "hlt8e29_trigNames" in alias: haveHLT8E29Info = True
                if "l1_trigNames" in alias: haveL1Info = True
                if "taus_pf_IDnames" in alias: haveTauIDInfo = True
                if "pfjets_bDiscriminatorNames" in alias: haveBtagInfo = True
        else:
            for br in temp_branches:
                branches.Add(br)

            
    if make_looper and (os.path.isfile("ScanChain.C") or os.path.isfile("doAll.C")):
        print(">>> Hey, you already have a looper here! I will be a bro and not overwrite them. Delete 'em if you want to regenerate 'em.")
        make_looper = False

    if make_looper:
        print(">>> Making looper")
        print(">>> Checking out cmstas/Software")
        os.system("[[ -d Software/ ]] || git clone https://github.com/cmstas/Software")

        buff = ""
        buff += "{\n"
        buff += "    gSystem->Exec(\"mkdir -p plots\");\n\n"
        buff += "    gROOT->ProcessLine(\".L Software/dataMCplotMaker/dataMCplotMaker.cc+\");\n"
        buff += "    gROOT->ProcessLine(\".L %s.cc+\");\n" % classname
        buff += "    gROOT->ProcessLine(\".L ScanChain.C+\");\n\n"
        buff += "    TChain *ch = new TChain(\"%s\");\n" % treename
        buff += "    ch->Add(\"%s\");\n\n" % fnames_in[0]
        buff += "    ScanChain(ch);\n\n"
        buff += "}\n\n"
        with open("doAll.C", "w") as fhout: fhout.write(buff)

        buff = ""
        buff += "#pragma GCC diagnostic ignored \"-Wsign-compare\"\n"
        buff += "#include \"Software/dataMCplotMaker/dataMCplotMaker.h\"\n\n"
        buff += "#include \"TFile.h\"\n"
        buff += "#include \"TTree.h\"\n"
        buff += "#include \"TCut.h\"\n"
        buff += "#include \"TColor.h\"\n"
        buff += "#include \"TCanvas.h\"\n"
        buff += "#include \"TH2F.h\"\n"
        buff += "#include \"TH1.h\"\n"
        buff += "#include \"TChain.h\"\n\n"
        buff += "#include \"%s.h\"\n\n" % classname
        buff += "using namespace std;\n"
        buff += "using namespace tas;\n\n"
        buff += "int ScanChain(TChain *ch){\n\n"
        buff += "    TH1F * h_met = new TH1F(\"met\", \"met\", 50, 0, 300);\n\n"
        buff += "    int nEventsTotal = 0;\n"
        buff += "    int nEventsChain = ch->GetEntries();\n\n"
        buff += "    TFile *currentFile = 0;\n"
        buff += "    TObjArray *listOfFiles = ch->GetListOfFiles();\n"
        buff += "    TIter fileIter(listOfFiles);\n\n"
        buff += "    while ( (currentFile = (TFile*)fileIter.Next()) ) { \n\n"
        buff += "        TFile *file = new TFile( currentFile->GetTitle() );\n"
        buff += "        TTree *tree = (TTree*)file->Get(\"%s\");\n" % treename
        buff += "        %s.Init(tree);\n\n" % objectname
        buff += "        TString filename(currentFile->GetTitle());\n\n"
        buff += "        for( unsigned int event = 0; event < tree->GetEntriesFast(); ++event) {\n\n"
        buff += "            %s.GetEntry(event);\n" % objectname
        buff += "            nEventsTotal++;\n\n"
        buff += "            %s::progress(nEventsTotal, nEventsChain);\n\n" % classname
        buff += "            h_met->Fill(evt_pfmet());\n\n"
        buff += "        }//event loop\n\n"
        buff += "        delete file;\n"
        buff += "    }//file loop\n\n"
        buff += "    TString comt = \" --outOfFrame --lumi 1.0 --type Simulation --darkColorLines --legendCounts --legendRight -0.05  --outputName plots/\";\n"
        buff += "    std::string com = comt.Data();\n"
        buff += "    TH1F * empty = new TH1F(\"\",\"\",1,0,1);\n\n"
        buff += "    dataMCplotMaker(empty,{h_met} ,{\"t#bar{t}\"},\"MET\",\"\",com+\"h_met.pdf --isLinear\");\n\n"
        buff += "    return 0;\n\n"
        buff += "}\n\n"
        with open("ScanChain.C", "w") as fhout: fhout.write(buff)

    d_bname_to_info = {}

    # cuts = ["filtcscBeamHalo2015","evtevent","evtlumiBlock","evtbsp4","hltprescales","hltbits","hlttrigNames","musp4","evtpfmet","muschi2","ak8jets_pfcandIndicies","hlt_prescales"]
    cuts = ["lep1_p4","lep2_p4"]
    isCMS3 = False
    have_aliases = False
    for branch in branches:
        bname = branch.GetName()
        cname = branch.GetClassName()
        btitle = branch.GetTitle()
        
        if bname in ["EventSelections", "BranchListIndexes", "EventAuxiliary", "EventProductProvenance"]: continue
        # if not any([cut in bname for cut in cuts]): continue

        # sometimes root is stupid and gives no class name, so must infer it from btitle (stuff like "btag_up/F")
        if not cname:
            if btitle.endswith("/i"): cname = "unsigned int"
            elif btitle.endswith("/l"): cname = "unsigned long long"
            elif btitle.endswith("/F"): cname = "float"
            elif btitle.endswith("/I"): cname = "int"
            elif btitle.endswith("/O"): cname = "bool"
            elif btitle.endswith("/D"): cname = "double"

        typ = cname[:]

        if "edm::TriggerResults" in cname:
            continue

        if "edm::Wrapper" in cname:
            isCMS3 = True
            typ = cname.replace("edm::Wrapper<","")[:-1]
        typ = classname_to_type(typ)

        if "musp4" in bname: print(bname)
        
        d_bname_to_info[bname] = {
                "class": cname,
                "alias": bname.replace(".",""),
                "type": typ,
                }        
        
    if len(aliases):
        have_aliases = True
        for iala, ala in enumerate(aliases):
            alias = ala.GetName()
            for tree in trees:
                if tree.GetBranch(tree.GetAlias(alias)):
                    branch = tree.GetBranch(tree.GetAlias(alias))
                    break
            # branchname = branch.GetName().replace("obj","")
            branchname = branch.GetName()
            if branchname not in d_bname_to_info: continue
            d_bname_to_info[branchname]["alias"] = alias.replace(".","")


    buff = ""

    ########################################
    ################ ***.h ################
    ########################################
    buff += '// -*- C++ -*-\n'
    buff += '#ifndef %s_H\n' % classname
    buff += '#define %s_H\n' % classname
    buff += '#include "Math/LorentzVector.h"\n'
    buff += '#include "Math/Point3D.h"\n'
    buff += '#include "TMath.h"\n'
    buff += '#include "TBranch.h"\n'
    buff += '#include "TTree.h"\n'
    buff += '#include "TH1F.h"\n'
    buff += '#include "TFile.h"\n'
    buff += '#include "TBits.h"\n'
    buff += '#include <vector>\n'
    buff += '#include <unistd.h>\n'
    buff += 'typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > LorentzVector;\n\n'

    buff += '// Generated with file: %s\n\n' % fnames_in[0]
    buff += 'using namespace std;\n'
    buff += 'class %s {\n' % classname
    buff += 'private:\n'
    buff += 'protected:\n'
    buff += '  unsigned int index;\n'
    for bname in d_bname_to_info:        
        alias = d_bname_to_info[bname]["alias"]
        typ = d_bname_to_info[bname]["type"]
        cname = d_bname_to_info[bname]["class"]
        needpointer = "vector<" in typ or "LorentzVector" in typ
        buff += '  %s %s%s_;\n' % (typ.replace("const","").strip(),"*" if needpointer else "",alias)
        buff += '  TBranch *%s_branch;\n' % (alias)
        buff += '  bool %s_isLoaded;\n' % (alias)
    buff += 'public:\n'
    buff += '  void Init(TTree *tree);\n'
    buff += '  void GetEntry(unsigned int idx);\n'
    buff += '  void LoadAllBranches();\n'
    for bname in d_bname_to_info:
        alias = d_bname_to_info[bname]["alias"]
        typ = d_bname_to_info[bname]["type"]
        buff += '  %s &%s();\n' % (typ, alias)
    if haveHLTInfo:
        buff += "  bool passHLTTrigger(TString trigName);\n"
    if haveHLT8E29Info:
        buff += "  bool passHLT8E29Trigger(TString trigName);\n"
    if haveL1Info:
        buff += "  bool passL1Trigger(TString trigName);\n"
    if haveTauIDInfo:
        buff += "  float passTauID(TString idName, unsigned int tauIndex);\n"
    if haveBtagInfo:
        buff += "  float getbtagvalue(TString bDiscriminatorName, unsigned int jetIndex);\n" 
    buff += "  static void progress( int nEventsTotal, int nEventsChain );\n"
    buff += '};\n\n'

    buff += "#ifndef __CINT__\n"
    buff += "extern %s %s;\n" % (classname, objectname)
    buff += "#endif\n"
    buff += "\n"
    buff += "namespace %s {\n" % (namespace)
    buff += "\n"
    for bname in d_bname_to_info:
        alias = d_bname_to_info[bname]["alias"]
        typ = d_bname_to_info[bname]["type"]
        buff += "  %s &%s();\n" % (typ, alias)
    if haveHLTInfo:
        buff += "  bool passHLTTrigger(TString trigName);\n"
    if haveHLT8E29Info:
        buff += "  bool passHLT8E29Trigger(TString trigName);\n"
    if haveL1Info:
        buff += "  bool passL1Trigger(TString trigName);\n"
    if haveTauIDInfo:
        buff += "  float passTauID(TString idName, unsigned int tauIndex);\n"
    if haveBtagInfo:
        buff += "  float getbtagvalue(TString bDiscriminatorName, unsigned int jetIndex);\n" 
    buff += "}\n"
    buff += "#endif\n"

    with open("%s.h" % classname, "w") as fhout:
        fhout.write(buff)

    print(">>> Saved %s.h" % (classname))

    ########################################
    ############### ***.cc ################
    ########################################

    buff = ""
    buff += '#include "%s.h"\n' % classname
    buff += "%s %s;\n\n" % (classname, objectname)
    buff += "void %s::Init(TTree *tree) {\n" % (classname)

    ## NOTE Do this twice. First time we consider branches for which we don't want to have SetMakeClass of 1
    for bname in d_bname_to_info:
        alias = d_bname_to_info[bname]["alias"]
        cname = d_bname_to_info[bname]["class"]
        if not(("TBits" in cname or "LorentzVector" in cname) and not "vector<vector" in cname): continue # NOTE
        buff += '  %s_branch = 0;\n' % (alias)
        if have_aliases:
            buff += '  if (tree->GetAlias("%s") != 0) {\n' % (alias)
            buff += '    %s_branch = tree->GetBranch(tree->GetAlias("%s"));\n' % (alias, alias)
        else:
            buff += '  if (tree->GetBranch("%s") != 0) {\n' % (alias)
            buff += '    %s_branch = tree->GetBranch("%s");\n' % (alias, alias)
        buff += '    if (%s_branch) { %s_branch->SetAddress(&%s_); }\n' % (alias, alias, alias)
        buff += '  }\n'

    buff += "  tree->SetMakeClass(1);\n"
    for bname in d_bname_to_info:
        alias = d_bname_to_info[bname]["alias"]
        cname = d_bname_to_info[bname]["class"]
        if ("TBits" in cname or "LorentzVector" in cname) and not "vector<vector" in cname: continue # NOTE
        buff += '  %s_branch = 0;\n' % (alias)
        if have_aliases:
            buff += '  if (tree->GetAlias("%s") != 0) {\n' % (alias)
            buff += '    %s_branch = tree->GetBranch(tree->GetAlias("%s"));\n' % (alias, alias)
        else:
            buff += '  if (tree->GetBranch("%s") != 0) {\n' % (alias)
            buff += '    %s_branch = tree->GetBranch("%s");\n' % (alias, alias)
        buff += '    if (%s_branch) { %s_branch->SetAddress(&%s_); }\n' % (alias, alias, alias)
        buff += '  }\n'

    buff += '  tree->SetMakeClass(0);\n'
    buff += "}\n"

    buff += "void %s::GetEntry(unsigned int idx) {\n" % classname
    buff += "  index = idx;\n"
    for bname in d_bname_to_info:
        alias = d_bname_to_info[bname]["alias"]
        buff += '  %s_isLoaded = false;\n' % (alias)
    buff += "}\n"

    buff += "void %s::LoadAllBranches() {\n" % classname
    for bname in d_bname_to_info:
        alias = d_bname_to_info[bname]["alias"]
        buff += '  if (%s_branch != 0) %s();\n' % (alias, alias)
    buff += "}\n"
    
    for bname in d_bname_to_info:
        alias = d_bname_to_info[bname]["alias"]
        typ = d_bname_to_info[bname]["type"]
        cname = d_bname_to_info[bname]["class"]
        needpointer = "vector<" in typ or "LorentzVector" in typ
        buff += "%s &%s::%s() {\n" % (typ, classname, alias)
        buff += "  if (not %s_isLoaded) {\n" % (alias)
        buff += "    if (%s_branch != 0) {\n" % (alias)
        buff += "      %s_branch->GetEntry(index);\n" % (alias)
        buff += "    } else {\n"
        buff += '      printf("branch %s_branch does not exist!\\n");\n' % (alias)
        buff += "      exit(1);\n"
        buff += "    }\n"
        buff += "    %s_isLoaded = true;\n" % (alias)
        buff += "  }\n"
        buff += "  return %s%s_;\n" % ("*" if needpointer else "", alias)
        buff += "}\n"

    if haveHLTInfo:
        buff += "bool %s::passHLTTrigger(TString trigName) {\n" % (classname)
        buff += "  int trigIndex;\n"
        buff += "  std::vector<TString>::const_iterator begin_it = hlt_trigNames().begin();\n"
        buff += "  std::vector<TString>::const_iterator end_it = hlt_trigNames().end();\n"
        buff += "  std::vector<TString>::const_iterator found_it = std::find(begin_it, end_it, trigName);\n"
        buff += "  if (found_it != end_it)\n"
        buff += "    trigIndex = found_it - begin_it;\n"
        buff += "  else {\n"
        buff += "    std::cout << \"Cannot find trigger \" << trigName << std::endl;\n"
        buff += "    return 0;\n"
        buff += "  }\n"
        buff += "  return hlt_bits().TestBitNumber(trigIndex);\n"
        buff += "}\n"

    if haveHLT8E29Info:
        buff += "bool %s::passHLT8E29Trigger(TString trigName) {\n" % (classname)
        buff += "  int trigIndex;\n"
        buff += "  std::vector<TString>::const_iterator begin_it = hlt8e29_trigNames().begin()\n;"
        buff += "  std::vector<TString>::const_iterator end_it = hlt8e29_trigNames().end();\n"
        buff += "  std::vector<TString>::const_iterator found_it = std::find(begin_it, end_it, trigName);\n"
        buff += "  if (found_it != end_it)\n"
        buff += "    trigIndex = found_it - begin_it;\n"
        buff += "  else {\n"
        buff += "    std::cout << \"Cannot find trigger \" << trigName << std::endl;\n"
        buff += "    return 0;\n"
        buff += "  }\n"
        buff += "  return hlt8e29_bits().TestBitNumber(trigIndex);\n"
        buff += "}\n"

    if haveL1Info:
        buff += "bool %s::passL1Trigger(TString trigName) {\n" % (classname)
        buff += "  int trigIndex;\n"
        buff += "  std::vector<TString>::const_iterator begin_it = l1_trigNames().begin();\n"
        buff += "  std::vector<TString>::const_iterator end_it = l1_trigNames().end();\n"
        buff += "  std::vector<TString>::const_iterator found_it = std::find(begin_it, end_it, trigName);\n"
        buff += "  if (found_it != end_it)\n"
        buff += "    trigIndex = found_it - begin_it;\n"
        buff += "  else {\n"
        buff += "    std::cout << \"Cannot find trigger \" << trigName << std::endl;\n"
        buff += "    return 0;\n"
        buff += "  }\n"
        list_of_aliases = []
        for iala, ala in enaliases:
            alias = ala.GetName()
            if "l1_bits" not in alias: continue
            for tree in trees:
                if tree.GetBranch(tree.GetAlias(alias)):
                    branch = tree.GetBranch(tree.GetAlias(alias))
                    break
            name_of_class = branch.GetClassName()
            if "int" not in name_of_class: continue
            list_of_aliases.append(alias)
        for iala, ala in enumerate(sorted(list_of_aliases)):
            if iala == 0:
                buff += "  if (trigIndex <= 31) {\n"
                buff += "    unsigned int bitmask = 1;\n"
                buff += "    bitmask <<= trigIndex;\n"
                buff += "    return %s() & bitmask;\n" % (ala)
                buff += "  }\n"
            else:
                buff += "  if (trigIndex >= %d && trigIndex <= %d) {\n" % (32*iala, 32+iala+31)
                buff += "    unsigned int bitmask = 1;\n"
                buff += "    bitmask <<= (trigIndex - %d);\n" % (32*iala)
                buff += "    return %s() & bitmask;\n" % (ala)
                buff += "  }\n"                
        buff += "  return 0;\n"        
        buff += "}\n"

    if haveTauIDInfo:
        buff += "float %s::passTauID(TString idName, unsigned int tauIndex) {\n" % (classname)
        buff += "  int idIndex;\n"
        buff += "  std::vector<TString>::const_iterator begin_it = taus_pf_IDnames().begin();\n"
        buff += "  std::vector<TString>::const_iterator end_it = taus_pf_IDnames().end();\n"
        buff += "  std::vector<TString>::const_iterator found_it = std::find(begin_it, end_it, idName);\n"
        buff += "  if (found_it != end_it)\n"
        buff += "    idIndex = found_it - begin_it;\n"
        buff += "  else {\n"
        buff += "    std::cout << \"Cannot find tau ID \" << idName << std::endl;\n"
        buff += "    return 0;\n"
        buff += "  }\n"
        buff += "  if (tauIndex < taus_pf_IDs().size())\n"
        buff += "    return taus_pf_IDs().at(tauIndex).at(idIndex);\n"
        buff += "  else {\n"
        buff += "    std::cout << \"Cannot find tau # \" << tauIndex << std::endl;\n"
        buff += "    return 0;\n"
        buff += "  }\n"
        buff += "}\n"        

    if haveBtagInfo:
        buff += "float %s::getbtagvalue(TString bDiscriminatorName, unsigned int jetIndex) {\n" % (classname)
        buff += "  size_t bDiscriminatorIndex;\n"
        buff += "  std::vector<TString>::const_iterator begin_it = pfjets_bDiscriminatorNames().begin();\n"
        buff += "  std::vector<TString>::const_iterator end_it = pfjets_bDiscriminatorNames().end();\n"
        buff += "  std::vector<TString>::const_iterator found_it = std::find(begin_it, end_it, bDiscriminatorName);\n"
        buff += "  if (found_it != end_it)\n"
        buff += "    bDiscriminatorIndex = found_it - begin_it;\n"
        buff += "  else {\n"
        buff += "    std::cout << \"Cannot find b-discriminator \" << bDiscriminatorName << std::endl;\n"
        buff += "    return 0;\n"
        buff += "  }\n"
        buff += "  if (jetIndex < pfjets_bDiscriminators().size())\n"
        buff += "    return pfjets_bDiscriminators().at(jetIndex).at(bDiscriminatorIndex);\n"
        buff += "  else {\n"
        buff += "    std::cout << \"Cannot find jet # \" << jetIndex << std::endl;\n"
        buff += "    return 0;\n"        
        buff += "  }\n"
        buff += "}\n"        
        
    buff += "void %s::progress( int nEventsTotal, int nEventsChain ){\n" % (classname)
    buff += "  int period = 1000;\n"
    buff += "  if(nEventsTotal%1000 == 0) {\n"
    buff += "    if (isatty(1)) {\n"
    buff += "      if( ( nEventsChain - nEventsTotal ) > period ){\n"
    buff += "        float frac = (float)nEventsTotal/(nEventsChain*0.01);\n"
    buff += "        printf(\"\\015\\033[32m ---> \\033[1m\\033[31m%4.1f%%\"\n"
    buff += "               \"\\033[0m\\033[32m <---\\033[0m\\015\", frac);\n"
    buff += "        fflush(stdout);\n"
    buff += "      }\n"
    buff += "      else {\n"
    buff += "        printf(\"\\015\\033[32m ---> \\033[1m\\033[31m%4.1f%%\"\n"
    buff += "               \"\\033[0m\\033[32m <---\\033[0m\\015\", 100.);\n"
    buff += "        cout << endl;\n"
    buff += "      }\n"
    buff += "    }\n"
    buff += "  }\n"
    buff += "}\n"

    buff += "namespace %s {\n" % (namespace)
    for bname in d_bname_to_info:
        alias = d_bname_to_info[bname]["alias"]
        typ = d_bname_to_info[bname]["type"]
        buff += "  %s &%s() { return %s.%s(); }\n" % (typ, alias, objectname, alias)
    if haveHLTInfo:
        buff += "  bool passHLTTrigger(TString trigName) { return %s.passHLTTrigger(trigName); }\n" % (objectname)
    if haveHLT8E29Info:
        buff += "  bool passHLT8E29Trigger(TString trigName) { return %s.passHLT8E29Trigger(trigName); }\n" % (objectname)
    if haveL1Info:
        buff += "  bool passL1Trigger(TString trigName) { return %s.passL1Trigger(trigName); }\n" % (objectname)
    if haveTauIDInfo:
        buff += "  float passTauID(TString idName, unsigned int tauIndex) { return %s.passTauID(idName, tauIndex); }\n" % (objectname)
    if haveBtagInfo:
        buff += "  float getbtagvalue(TString bDiscriminatorName, unsigned int jetIndex) { return %s.getbtagvalue(bDiscriminatorName, jetIndex); }\n" % (objectname)
    buff += "}\n"

    with open("%s.cc" % classname, "w") as fhout:
        fhout.write(buff)
    print(">>> Saved %s.cc" % (classname))



