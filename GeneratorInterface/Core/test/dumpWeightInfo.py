from __future__ import print_function
from DataFormats.FWLite import Events,Handle,Runs,Lumis
import ROOT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str, help="Input EDM file")
parser.add_argument("--source", type=str, help="product ID of weight product", default="externalLHEProducer")
args = parser.parse_args()

lumis = Lumis(args.infile)
lumi = lumis.__iter__().next()
weightInfoHandle = Handle("GenWeightInfoProduct")
print("Trying to get weightInfo from lumi")
lumi.getByLabel(args.source, weightInfoHandle)
weightInfoProd = weightInfoHandle.product()

events = Events(args.infile)
event = events.__iter__().next()
weightHandle = Handle("GenWeightProduct")
print("Trying to get weightProduct from event")
event.getByLabel(args.source, weightHandle)
weightInfo = weightHandle.product()
print("Number of weight groups in weightInfo is", len(weightInfo.weights()))
for j, weights in enumerate(weightInfo.weights()):
    print("-"*10, "Looking at entry", j, "length is", len(weights),"-"*10)
    matching = weightInfoProd.orderedWeightGroupInfo(j)
    print("Group is", matching, "name is", matching.name(), "well formed?", matching.isWellFormed())
    print("Group description", matching.description())
    print(type(matching.weightType()), matching.weightType())
    if matching.weightType() == 's':
        for var in [(x, y) for x in ["05", "1", "2"] for y in ["05", "1", "2"]]:
            name = "muR%smuF%sIndex" % (var[0], var[1]) if not (var[0] == "1" and var[1] == "1") else "centralIndex"
            print(name, getattr(matching, name)())
    elif matching.weightType() == 'P':
        print("uncertaintyType", "Hessian" if matching.uncertaintyType() == ROOT.gen.kHessianUnc else "MC")
        print("Has alphas? ", matching.hasAlphasVariations())
    print("Weights length?", len(weights), "Contained ids lenths?", len(matching.containedIds()))
    print("-"*80)
    for i,weight in enumerate(weights):
        print(i, weight)
        info = matching.weightMetaInfo(i)
        print("   ID, localIndex, globalIndex, label, set:", info.id, info.localIndex, info.globalIndex, info.label, matching.name())
    print("-"*80)
