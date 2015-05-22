from CMGTools.RootTools.PyRoot import * 
from CMGTools.RootTools.Style import * 
from CMGTools.RootTools.HistComparator import * 

num = 0
def hname():
    global num
    num+=1
    return 'h_{num}'.format(num=num)

legend = None
tree1 = None
tree2 = None
a1 = None
a2 = None

def draw(var1=None, cut=1, t1=None, t2=None, w1='1', w2='1',
         name1=None, name2=None,
         normalize=None, nbins=20, xmin=0, xmax=200, var2=None):
    if var2 is None:
        var2 = var1
    if t1 is None:
        t1 = tree1
    if t2 is None:
        t2 = tree2
    if name1 is None:
        name1 = a1
    if name2 is None:
        name2 = a2
    print 'tree1',
    print '\t var   : ' , var1
    print '\t weight:', w1
    print 'tree2',
    print '\t var   : ' , var2
    print '\t weight:', w2
    print 'cut', cut
    global legend
    h1 = TH1F(hname(), '', nbins, xmin, xmax)
    h1.Sumw2()
    t1.Project(h1.GetName(), var1,'({cut})*({w1})'.format(cut=cut,w1=w1),'')
    h2 = h1.Clone(hname())
    h2.Sumw2()
    t2.Project(h2.GetName(), var2,'({cut})*({w2})'.format(cut=cut,w2=w2),'')
    if normalize == None:
        pass
    elif normalize == -1:
        h1.Scale(1./h1.Integral())
        h2.Scale(1./h2.Integral())
    elif normalize>0:
        h2.Scale( normalize )
    sBlue.markerStyle = 25
    sBlue.formatHisto(h2)
    sData.formatHisto(h1)
    h2.SetFillStyle(1001)
    h2.SetFillColor(5)
    h1.SetMarkerSize(0.8)
    h2.SetMarkerSize(0.8)
    h1.SetStats(0)
    h2.SetStats(0)
    if name1 is None: name1 = t1.GetTitle()
    if name2 is None: name2 = t2.GetTitle()
    h1.SetTitle(name1)
    h2.SetTitle(name2)
    legend = TLegend(0.55,0.7,0.88,0.88)
    legend.SetFillColor(0)
    legend.AddEntry(h1, name1, 'lp')
    legend.AddEntry(h2, name2, 'lpf')

    print 'number of selected rows:', t1.GetSelectedRows(), t2.GetSelectedRows()

    comparator = HistComparator(var1, h1, h2)
    comparator.draw(opt2='e2')
    return comparator


def simpleDraw(var, cut='1'):
    t1 = tree1
    t2 = tree2
    name1 = tree1.GetName()
    name2 = tree2.GetName()

    return draw(var1=var, cut=cut, t1=t1, t2=t2, name1=name1, name2=name2)
    

def getTreesOld( treeName, patterns ):
    trees = dict()
    for alias, pattern in patterns:
        print 'loading', alias, treeName, pattern
        tree = Chain(treeName, pattern)
        tmpalias = alias
        num=0
        while tmpalias in trees:
            num += 1
            tmpalias = '{alias}_{num}'.format(alias=alias, num=num)
        trees[tmpalias] = tree
        # tree.SetWeight(1./tree.GetEntries(), 'global')
    return trees

def getTree(arg):
    name = None
    filepattern = arg
    if ':' in arg:
        name, filepattern = arg.split(':')
    tree = Chain(name, filepattern)
    return tree


def main():
    import sys
    import pprint
    from optparse import OptionParser
    
    parser = OptionParser()
    
    parser.usage = """
    %prog -i <tree_alias:root_file_name> <tree_alias:root_file_name>

    if you do not provide the var option, you can e.g. do:
    comp = draw('jet2_eta', 'jet2_pt>30', trees[a1], trees[a2], name1=a1, name2=a2, xmin=-5, xmax=5); comp.draw()
    """
    parser.add_option("-v", "--var", 
                      dest="var", 
                      help="variable to draw.",
                      default=None)
    parser.add_option("-c", "--cut", 
                      dest="cut", 
                      help="cut to apply",
                      default='1')
    parser.add_option("-o", "--outdir", 
                      dest="outdir", 
                      help="output director for plots",
                      default='Comparator_OutDir')
    parser.add_option("-t", "--tree", 
                      dest="tree", 
                      help="name of tree in files",
                      default=None)
    parser.add_option("-1", "--alias1", 
                      dest="alias1", 
                      help="alias for the first tree",
                      default=None)
    parser.add_option("-2", "--alias2", 
                      dest="alias2", 
                      help="alias for the second tree",
                      default=None)
    
    

    (options,args) = parser.parse_args()

    if len(args)!=2:
        parser.print_usage()
        sys.exit(1)

    global tree1, tree2, a1, a2
    
    tree1 = getTree(args[0])
    a1 = options.alias1
    if a1 is None:
        a1 = tree1.GetName()
    tree2 = getTree(args[1])
    a2 = options.alias2
    if a2 is None:
        a2 = tree2.GetName()
        
    comp = None
    if options.var:
        comp = draw(options.var, options.cut,
                    tree1, tree2,
                    name1=a1, name2=a2);
        comp.draw()
        
    return tree1, tree2, options, comp

if __name__ == '__main__':
    tree1, tree2, options, comparator = main() 
