#!/bin/env python
import ROOT
import numpy
import sys

if len(sys.argv) < 3:
    print("Syntax: haddnano.py out.root input1.root input2.root ...")
ofname = sys.argv[1]
files = sys.argv[2:]


def zeroFill(tree, brName, brObj, allowNonBool=False):
    # typename: (numpy type code, root type code)
    branch_type_dict = {'Bool_t': ('?', 'O'), 'Float_t': ('f4', 'F'), 'UInt_t': (
        'u4', 'i'), 'Long64_t': ('i8', 'L'), 'Double_t': ('f8', 'D')}
    brType = brObj.GetLeaf(brName).GetTypeName()
    if (not allowNonBool) and (brType != "Bool_t"):
        print(("Did not expect to back fill non-boolean branches", tree, brName, brObj.GetLeaf(br).GetTypeName()))
    else:
        if brType not in branch_type_dict:
            raise RuntimeError('Impossible to backfill branch of type %s' % brType)
        buff = numpy.zeros(1, dtype=numpy.dtype(branch_type_dict[brType][0]))
        b = tree.Branch(brName, buff, brName + "/" +
                        branch_type_dict[brType][1])
        # be sure we do not trigger flushing
        b.SetBasketSize(tree.GetEntries() * 2)
        for x in range(0, tree.GetEntries()):
            b.Fill()
        b.ResetAddress()


fileHandles = []
goFast = True
for fn in files:
    print("Adding file" + str(fn))
    fileHandles.append(ROOT.TFile.Open(fn))
    if fileHandles[-1].GetCompressionSettings() != fileHandles[0].GetCompressionSettings():
        goFast = False
        print("Disabling fast merging as inputs have different compressions")
of = ROOT.TFile(ofname, "recreate")
if goFast:
    of.SetCompressionSettings(fileHandles[0].GetCompressionSettings())
of.cd()

for e in fileHandles[0].GetListOfKeys():
    name = e.GetName()
    print("Merging" + str(name))
    obj = e.ReadObj()
    cl = ROOT.TClass.GetClass(e.GetClassName())
    inputs = ROOT.TList()
    isTree = obj.IsA().InheritsFrom(ROOT.TTree.Class())
    if isTree:
        obj = obj.CloneTree(-1, "fast" if goFast else "")
        branchNames = set([x.GetName() for x in obj.GetListOfBranches()])
    for fh in fileHandles[1:]:
        otherObj = fh.GetListOfKeys().FindObject(name).ReadObj()
        inputs.Add(otherObj)
        if isTree and obj.GetName() == 'Events':
            otherObj.SetAutoFlush(0)
            otherBranches = set([x.GetName()
                                 for x in otherObj.GetListOfBranches()])
            missingBranches = list(branchNames - otherBranches)
            additionalBranches = list(otherBranches - branchNames)
            print("missing: " + str(missingBranches) + "\n Additional:" + str(additionalBranches))
            for br in missingBranches:
                # fill "Other"
                zeroFill(otherObj, br, obj.GetListOfBranches().FindObject(br))
            for br in additionalBranches:
                # fill main
                branchNames.add(br)
                zeroFill(obj, br, otherObj.GetListOfBranches().FindObject(br))
            # merge immediately for trees
        if isTree and obj.GetName() == 'Runs':
            otherObj.SetAutoFlush(0)
            otherBranches = set([x.GetName()
                                 for x in otherObj.GetListOfBranches()])
            missingBranches = list(branchNames - otherBranches)
            additionalBranches = list(otherBranches - branchNames)
            print("missing: " + str(missingBranches) + "\n Additional:" + str(additionalBranches))
            for br in missingBranches:
                # fill "Other"
                zeroFill(otherObj, br, obj.GetListOfBranches(
                ).FindObject(br), allowNonBool=True)
            for br in additionalBranches:
                # fill main
                branchNames.add(br)
                zeroFill(obj, br, otherObj.GetListOfBranches(
                ).FindObject(br), allowNonBool=True)
            # merge immediately for trees
        if isTree:
            obj.Merge(inputs, "fast" if goFast else "")
            inputs.Clear()

    if isTree:
        obj.Write()
    elif obj.IsA().InheritsFrom(ROOT.TH1.Class()):
        obj.Merge(inputs)
        obj.Write()
    elif obj.IsA().InheritsFrom(ROOT.TObjString.Class()):
        for st in inputs:
            if st.GetString() != obj.GetString():
                print("Strings are not matching")
        obj.Write()
    else:
        print("Cannot handle " + str(obj.IsA().GetName()))
