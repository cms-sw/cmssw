import copy

from ROOT import TChain, TH1F
from CMGTools.H2TauTau.proto.plotter.categories_TauMu import cat_Inc
from CMGTools.H2TauTau.proto.plotter.categories_TauEle import cat_Inc as cat_Inc_Ele


def embedScaleFactor(selComps, channel='TauMu'):
    embedChain = TChain()
    zttChain = None
    for comp in selComps.values():
        if comp.isEmbed:
            #if embedChain is None:
            #    embedChain = copy.deepcopy( comp.tree )
            #    continue
            # import pdb; pdb.set_trace()
            print 'chaining', comp.name, comp.tree.GetEntries()
            embedChain.Add(comp.tree)
        elif comp.name=='Ztt':
            zttChain = comp.tree
    cut = cat_Inc 
    if channel == 'TauEle':
        cut = cat_Inc_Ele
    cut += ' && diTau_charge==0'
    embcut = cut 
    zttcut = ' && '.join( [cut, 'isFake==0'] )
    eh = TH1F('eh','eh', 100, 0, 2000)
    zh = TH1F('zh','zh', 100, 0, 2000)
    zh.SetLineColor(4)
    eh.Sumw2()
    zh.Sumw2()
    embedChain.Project('eh', 'visMass', '({cut})*weight'.format(cut=embcut))
    zttChain.Project('zh', 'visMass', '({cut})*weight'.format(cut=zttcut))
    factor = zh.Integral()/eh.Integral()

    print 'Integral Z', zh.Integral()
    print 'Integral emb', eh.Integral()
    return eh, zh, factor


def setupEmbedding(plot, doEmbedding, scaleFactor ):
    name = 'Ztt'
    # import pdb; pdb.set_trace()
    try:
        dyHist = plot.Hist(name)
    except KeyError:
        return 
    # dyYield = dyHist.Yield()

    # newName = 'Ztt (emb)'
    newName = name

    embed = None
    for comp in plot.selComps.values():
        if not comp.isEmbed:
            continue
        embedHistName = comp.name
        embedHist = plot.Hist( embedHistName )
        embedHist.stack = False
        embedHist.on = False
        if doEmbedding:
            if embed is None:
                embed = copy.deepcopy( embedHist )
                embed.name = 'Ztt'
                embed.on = True
                # plot.AddHistogram(newName, embed.weighted, 3.5)
                plot.Replace('Ztt', embed)
                plot.Hist(newName).stack = True
            else:
                plot.Hist(newName).Add(embedHist)
    if doEmbedding:
        #         embedYield = plot.Hist(newName).Yield()
        print 'EMBEDDING: scale factor = ', scaleFactor
        plot.Hist(newName).Scale( scaleFactor ) 
        plot._ApplyPrefs()
        # plot.Hist(name).on = False
