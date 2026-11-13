import json
import sys
import xml.etree.ElementTree as ET
import ROOT

ROOT.gROOT.SetBatch(True)
THING = 'edmtestThings_alpha__LAYOUT.'
INT = 'edmtestIntProduct_beta__LAYOUT.'
BLOCK = 'edmtestIntProduct_block__LAYOUT.'
PRODUCTS = ['edmTriggerResults_TriggerResults__LAYOUT.', INT, THING]

def branch(tree, name):
    result = tree.GetBranch(name)
    assert result, (tree.GetName(), name)
    return result

def properties(tree, name):
    b = branch(tree, name)
    return [b.GetSplitLevel(), b.GetBasketSize()]

def snapshot(filename):
    f = ROOT.TFile.Open(filename)
    assert f and not f.IsZombie(), filename
    tree = f.Get('Events')
    assert tree
    result = {
        'compression': f.GetCompressionSettings(),
        'order': [b.GetName() for b in tree.GetListOfBranches() if b.GetName() in PRODUCTS],
        'aux': properties(tree, 'EventAuxiliary'),
        'top': properties(tree, THING),
        'int': properties(tree, INT),
        'nested': properties(tree, f'{THING}obj'),
        'deep': properties(tree, f'{THING}obj.a'),
    }
    f.Close()
    return result

def values(tree, name, expected):
    b = branch(tree, name)
    assert b.GetEntries() == tree.GetEntries(), name
    leaf = b.GetListOfLeaves().At(0)
    assert leaf, name
    for i in range(tree.GetEntries()):
        assert tree.GetEntry(i) > 0, (tree.GetName(), i)
        assert [leaf.GetValue(j) for j in range(leaf.GetLen())] == expected, (name, i)

def check_values(filename, nfiles):
    f = ROOT.TFile.Open(filename)
    assert f and not f.IsZombie(), filename

    def checked_tree(name, count):
        tree = f.Get(name)
        assert tree and tree.GetEntries() == count, (filename, name, count)
        return tree

    events = checked_tree('Events', 6 * nfiles)
    values(events, f'{THING}obj.a', [0, 1, 2, 3])
    values(events, f'{INT}obj.value', [42])

    lumis = checked_tree('LuminosityBlocks', 2 * nfiles)
    for transition in ('beginLumi', 'endLumi'):
        values(lumis, f'edmtestThings_alpha_{transition}_LAYOUT.obj.a', [0, 1, 2, 3])

    runs = checked_tree('Runs', nfiles)
    for transition in ('beginRun', 'endRun'):
        values(runs, f'edmtestThings_alpha_{transition}_LAYOUT.obj.a', [0, 1, 2, 3])

    process_blocks = checked_tree('ProcessBlocksLAYOUT', nfiles)
    values(process_blocks, f'{BLOCK}obj.value', [73])
    f.Close()

def fast_copy(mode, expected):
    report = ET.parse(f'layout_{mode}.xml')
    output = report.find('./File')
    assert output is not None
    actual = {item.findtext('PFN'): item.findtext('FastCopying') for item in output.findall('./Inputs/Input')}
    assert actual == expected, (mode, actual, expected)

def prepare():
    for name in ('layout_input.root', 'layout_input2.root'):
        f = ROOT.TFile.Open(name, 'UPDATE')
        assert f and not f.IsZombie()
        tree = f.Get('Events')
        top = branch(tree, THING)
        assert top.GetSplitLevel() > 1
        branch(tree, f'{THING}obj').SetBasketSize(28672)
        branch(tree, f'{THING}obj.a').SetBasketSize(36864)
        tree.Write('', ROOT.TObject.kOverwrite)
        f.Close()

def record():
    first, second = (snapshot(n) for n in ('layout_input.root', 'layout_input2.root'))
    assert first['compression'] == second['compression'] == 101
    assert first['order'] == second['order'] == PRODUCTS
    assert first['aux'] == [0, 12288]
    assert first['top'] == [99, 16384] and second['top'] == [99, 24576]
    assert first['int'] == [99, 16384] and second['int'] == [99, 24576]
    assert first['nested'] == [98, 28672] and first['deep'] == [0, 36864]
    assert second['nested'] == first['nested'] and second['deep'] == first['deep']
    check_values('layout_input.root', 1)
    check_values('layout_input2.root', 1)
    with open('layout_input_properties.json', 'w') as output:
        json.dump([first, second], output, indent=2)
    print('Input properties:', first, second)

def check():
    with open('layout_input_properties.json') as source:
        first, second = json.load(source)
    for mode in ('clone', 'copy', 'merge', 'override', 'overridefast'):
        actual = snapshot(f'layout_{mode}.root')
        assert actual['order'] == first['order'], (mode, actual)
        assert actual['compression'] == (101 if mode == 'merge' else 505), (mode, actual)
        if mode not in ('override', 'overridefast'):
            for key in ('aux', 'top', 'int'):
                assert actual[key] == first[key], (mode, key, actual, first)
        else:
            assert actual['aux'] == [0, 32768] and actual['top'] == first['top'], (mode, actual)
            assert actual['int'] == [99, 16384], (mode, actual)
        if mode == 'merge':
            for key in ('nested', 'deep'):
                assert actual[key] == first[key], (mode, key, actual, first)
        else:
            assert actual['nested'] == [98, 16384] and actual['deep'] == [0, 16384], (mode, actual)
        check_values(f'layout_{mode}.root', 2 if mode.startswith('override') else 1)
    one = {'file:layout_input.root': '1'}
    zero = {'file:layout_input.root': '0'}
    fast_copy('clone', one)
    fast_copy('copy', zero)
    fast_copy('merge', zero)
    fast_copy('override', {**zero, 'file:layout_input2.root': '0'})
    fast_copy('overridefast', {**one, 'file:layout_input2.root': '0'})


def main():
    actions = {'prepare': prepare, 'record': record, 'check': check}
    if len(sys.argv) != 2:
        raise SystemExit(f'Usage: {sys.argv[0]} <prepare|record|check>')
    action = sys.argv[1]
    if action not in actions:
        raise SystemExit(f"Unknown action '{action}'; expected prepare, record, or check")
    actions[action]()


if __name__ == '__main__':
    main()
