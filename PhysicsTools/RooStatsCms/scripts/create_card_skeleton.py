#! /usr/bin/env python

import sys
import os

'''
Create a card skeleton from commandline requests
'''

card_snippets_names=["combinedModel",
                     "singleModel",
                     "sigTopLevel",
                     "bkgTopLevel",
                     "bkgTopLevelCompositeYield",
                     "sigDeclaration",
                     "bkgDeclaration",
                     "bkgDeclarationMultipleComponents"]

if len(sys.argv)!=2:
   print "Usage %s snippet_name\n" %os.path.basename(sys.argv[0])
   print "Choose a snippet_name among:"
   for name in card_snippets_names:
       print " - "+name
   print "\n"
   sys.exit(1)

snippet_name=sys.argv[1]

if snippet_name== "combinedModel":
    print "[<combinedModelName>]\n"+\
          "    model = combined\n"+\
          "    components = <singleModelName1>, <singleModelName2>, <singleModelName3>"


elif snippet_name=="singleModel":
    print "[<modelName>]\n"+\
          "    variables = <varname>\n"+\
          "    <varname> = 100 L(30 - 250)  // [GeV/c^{2}]"


elif snippet_name=="sigTopLevel":
    print "[<modelName>_sig]\n"+\
          "    number_components = 1\n"+\
          "    <modelName>_sig_yield = 3 C"

elif snippet_name=="bkgTopLevel":
    print "[<modelName>_bkg]\n"+\
          "    number_components = 1\n"+\
          "    <modelName>_bkg_yield = 10 C"

elif snippet_name=="bkgTopLevelCompositeYield":
    print "[<modelName>_bkg]\n"+\
          "    yield_factors_number = 2\n\n"+\
          "    yield_factor_1 = <name1>\n"+\
          "    <name1> = 1 L (0 - 20)\n"+\
          "    yield_factor_2 = <name2>\n"+\
          "    <name2> = 23.5 C"

elif snippet_name=="sigDeclaration":
    print "[<modelName>_sig_<varname>]\n"+\
          "    model = gauss\n"+\
          "    <modelName>_sig_<varname>_mean  = 140.702 C\n"+\
          "    <modelName>_sig_<varname>_sigma = 12.8216 C"

elif snippet_name=="bkgDeclaration":
    print "[<modelName>_bkg_<varname>]\n"+\
          "    model = gauss\n"+\
          "    <modelName>_bkg_<varname>_mean  = 140.702 C\n"+\
          "    <modelName>_bkg_<varname>_sigma = 12.8216 C"


elif snippet_name=="bkgDeclarationMultipleComponents":
    print "[<modelName>_bkg1_<varName>]\n"+\
          "model = BreitWigner\n"+\
          "<modelName>_bkg1_<varName>_mean  = 97.165 C\n"+\
          "<modelName>_bkg1_<varName>_width = 17.1847 C\n\n"+\
          "[<modelName>_bkg2_<varName>]\n"+\
          "model = poly7\n"+\
          "<modelName>_bkg2_<varName>_coef1 = -6.1666e+10 C\n"+\
          "<modelName>_bkg2_<varName>_coef2 =  3.0537e+09 C\n"+\
          "<modelName>_bkg2_<varName>_coef3 = -2.1112e+07 C\n"+\
          "<modelName>_bkg2_<varName>_coef4 =  6.3835e+04 C\n"+\
          "<modelName>_bkg2_<varName>_coef5 = -9.8900e+01 C\n"+\
          "<modelName>_bkg2_<varName>_coef6 =  7.4581e-02 C\n"+\
          "<modelName>_bkg2_<varName>_coef7 = -1.9461e-05 C\n"
          
else:
    print "%s: option not recognised! Please check the spelling." %sys.argv[1]
