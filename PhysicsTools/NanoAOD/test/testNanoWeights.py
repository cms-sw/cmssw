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
parser.add_argument('--scan', action='store_true', help='Scan the weight values')
args = parser.parse_args()

rtfile = ROOT.TFile(args.inputFile)
tree = rtfile.Get("Events")
tree.GetEntry(0)
variables = ["LHEScaleWeight", "LHEPdfWeight", "MEParamWeight", "UnknownWeight", "PSWeight", ]

for varName in variables:
    variableAndNumber(varName, tree)

if args.scan:
    tree.Scan(":".join(variables))
