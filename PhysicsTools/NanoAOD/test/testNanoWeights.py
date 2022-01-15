import ROOT
import argparse

def variableAndNumber(varName, tree):
    countVar = "n"+varName
    if not hasattr(tree, varName):
        print("No variable %s found in file" % varName)
    else:
        count = getattr(tree, countVar)
        var = getattr(tree, varName)
        print("Found %i entries of %s in file" % (count, varName))
        branch = tree.GetBranch(varName)
        print("    --> Desciption:%s" % branch.GetTitle())

parser = argparse.ArgumentParser()
parser.add_argument('inputFile', type=str, help='NanoAOD file to process')
args = parser.parse_args()

rtfile = ROOT.TFile(args.inputFile)
tree = rtfile.Get("Events")
tree.GetEntry(0)
types = ["ScaleWeight", "PdfWeight", "MEParamWeight", "UnknownWeight", ]
variables = ["LHE"+t for t in types]
variables.append("GenPartonShowerWeight")
variables.extend(["Gen"+t for t in types])

for varName in variables:
    variableAndNumber(varName, tree)
    i = 1
    altName = varName + "AltSet%i" % i
    while hasattr(tree, altName):
        variableAndNumber(altName, tree)
        i = i+1
        altName = varName + "AltSet%i" % i
