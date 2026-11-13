import sys
import xml.etree.ElementTree as ET
import ROOT

def check_report(path, output, statuses, read_count):
    report = ET.parse(path).getroot()
    outputs = [f for f in report.findall('.//File') if output in (f.findtext('PFN') or '')]
    assert len(outputs) == 1, 'expected one output file in JobReport'
    inputs = outputs[0].findall('./Inputs/Input')
    assert len(inputs) == 3, 'expected three contributing input files'
    status = {entry.findtext('PFN').split('/')[-1].removeprefix('file:'): entry.findtext('FastCopying')
              for entry in inputs}
    assert status == {'transitionInput%d.root' % i: value for i, value in enumerate(statuses, 1)}, status

    read_branches = report.findall('./ReadBranches/Branch')
    counts = {branch.get('Name'): int(branch.get('ReadCount')) for branch in read_branches}
    assert len(counts) == len(read_branches), 'duplicate ReadBranches entries'
    # JobReport combines regular reads and clone reports. Clone reports add
    # cumulative output entries, not just the current input's entries.
    for name in ('parent', 'child', 'specialParent', 'special'):
        branch = 'edmtestIntProduct_%s__INPUT.' % name
        assert counts.get(branch) == read_count, (path, branch, counts.get(branch), read_count)


check_report(sys.argv[1], 'transitionOutput.root', ('1', '1', '1'), 12)
check_report(sys.argv[2], 'transitionOutputNoClone.root', ('0', '0', '0'), 6)

root_file = ROOT.TFile.Open('transitionOutput.root')
assert root_file and not root_file.IsZombie()
tree = root_file.Get('Events')
assert tree and tree.GetEntries() == 6
for name in ('edmtestIntProduct_parent__INPUT.', 'edmtestIntProduct_child__INPUT.',
             'edmtestIntProduct_specialParent__INPUT.', 'edmtestIntProduct_special__INPUT.'):
    branch = tree.GetBranch(name)
    assert branch and branch.GetEntries() == 6, (name, branch.GetEntries() if branch else None)

# Status is reported before TTreeCloner runs. Compare the persisted payload of
# a split data basket as well, excluding the ROOT key header and its offsets.
def payload(file_name, branch_name, entry):
    f = ROOT.TFile.Open(file_name)
    branch = f.Get('Events').GetBranch(branch_name)
    assert branch
    candidates = [i for i in range(branch.GetMaxBaskets()) if branch.GetBasketSeek(i)]
    matching = [i for i in candidates if branch.GetBasketEntry()[i] == entry]
    assert len(matching) == 1, (file_name, branch_name, entry, candidates)
    index = matching[0]
    basket = branch.GetBasket(index)
    key_length = basket.GetKeylen()
    size = branch.GetBasketBytes()[index] - key_length
    assert size > 0
    with open(file_name, 'rb') as stream:
        stream.seek(branch.GetBasketSeek(index) + key_length)
        result = stream.read(size)
    assert len(result) == size
    return result


for name in ('parent', 'child', 'specialParent', 'special'):
    branch_name = 'edmtestIntProduct_%s__INPUT.obj.value' % name
    assert payload('transitionInput1.root', branch_name, 0) == payload('transitionOutput.root', branch_name, 0)
    assert payload('transitionInput3.root', branch_name, 0) == payload('transitionOutput.root', branch_name, 4)
