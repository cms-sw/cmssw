#!/usr/bin/env python

import re
import sys
import array

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True


colors = [
    ROOT.kBlue,
    ROOT.kRed+1,
    ROOT.kBlack
]

def findBounds(x, ys, xmin=None, ymin=None, xmax=None, ymax=None):
    if xmin is None:
        xmin = min(x)
    if xmax is None:
        xmax = max(x)
    if ymin is None:
        ymin = min([min(y) for y in ys])
    if ymax is None:
        ymax = max([max(y) for y in ys]) * 1.1

    return (xmin, ymin, xmax, ymax)


def makePlot(name, x, ys, ytitle,
             title=None,
             legends=None,
             ideal1=None,
             bounds={},
             legendYmax=0.99
         ):
    canv = ROOT.TCanvas()
    canv.cd()
    canv.SetTickx(1)
    canv.SetTicky(1)
    canv.SetGridy(1)

    bounds = findBounds(x, ys, **bounds)
    frame = canv.DrawFrame(*bounds)

    frame.GetXaxis().SetTitle("Number of threads")
    frame.GetYaxis().SetTitle(ytitle)
    if title is not None:
        frame.SetTitle(title)
    frame.Draw("")

    leg = None
    if legends is not None:
        leg = ROOT.TLegend(0.77,legendYmax-0.19,0.99,legendYmax)

    graphs = []

    if ideal1 is not None:
        ymax = bounds[3]
        ideal_y = [ideal1, ymax]
        ideal_x = [1, ymax/ideal1]
        gr = ROOT.TGraph(2, array.array("d", ideal_x), array.array("d", ideal_y))
        gr.SetLineColor(ROOT.kBlack)
        gr.SetLineStyle(3)
        gr.Draw("same")
        if leg:
            leg.AddEntry(gr, "Ideal scaling", "l")
        graphs.append(gr)

    for i, y in enumerate(ys):
        gr = ROOT.TGraph(len(x), array.array("d", x), array.array("d", y))
        color = colors[i]
        gr.SetLineColor(color)
        gr.SetMarkerColor(color)
        gr.SetMarkerStyle(ROOT.kFullCircle)
        gr.SetMarkerSize(1)

        gr.Draw("LP SAME")
        if leg:
            leg.AddEntry(gr, legends[i], "lp")

        graphs.append(gr)

    if leg:
        leg.Draw("same")

    canv.SaveAs(name+".png")
    canv.SaveAs(name+".pdf")


def main(argv):
    (inputfile, outputfile, graph_label) = argv[1:4]

    re_mt = re.compile("nTH(?P<th>\d+)_nEV(?P<ev>\d+)")
    re_mp = re.compile("nJOB(?P<job>\d+)")

    mt = {}
    mp = {}

    f = open(inputfile)
    for line in f:
        if not "AVX512" in line:
            continue
        comp = line.split(" ")
        m = re_mt.search(comp[0])
        if m:
            if m.group("th") != m.group("ev"):
                raise Exception("Can't handle yet different numbers of threads (%s) and events (%s)" % (m.group("th"), m.group("ev")))
            mt[int(m.group("th"))] = float(comp[1])
            continue
        m = re_mp.search(comp[0])
        if m:
            mp[int(m.group("job"))] = float(comp[1])
    f.close()

    ncores = sorted(list(set(mt.keys() + mp.keys())))
    mt_y = [mt[n] for n in ncores]
    mp_y = [mp[n] for n in ncores]
    ideal1 = mt_y[0]/ncores[0]
    ideal1_mp = mp_y[0]/ncores[0]

    makePlot(outputfile+"_throughput", ncores,
             [mt_y, mp_y],
             "Throughput (events/s)",
             title=graph_label,
             legends=["Multithreading", "Multiprocessing"],
             ideal1=ideal1,
             bounds=dict(ymin=0, xmin=0),
             legendYmax=0.5
    )

    eff = [mt_y[i]/mp_y[i] for i in xrange(0, len(ncores))]
    makePlot(outputfile+"_efficiency", ncores,
             [eff],
             "Multithreading efficiency (MT/MP)",
             title=graph_label,
             bounds=dict(ymin=0.9, ymax=1.1)
    )

    eff_vs_ideal_mt = [mt_y[i]/(ideal1*n) for i, n in enumerate(ncores)]
    eff_vs_ideal_mp = [mp_y[i]/(ideal1*n) for i, n in enumerate(ncores)]
    makePlot(outputfile+"_efficiency_ideal", ncores,
             [eff_vs_ideal_mt, eff_vs_ideal_mp],
             "Efficiency wrt. ideal",
             title=graph_label,
             legends=["Multithreading", "Multiprocessing"],
             bounds=dict(ymin=0.8, ymax=1.01, xmax=65),
             legendYmax=0.9
    )

    speedup_mt = [mt_y[i]/ideal1 for i in xrange(0, len(ncores))]
    speedup_mp = [mp_y[i]/ideal1 for i in xrange(0, len(ncores))]
    makePlot(outputfile+"_speedup", ncores,
             [speedup_mt, speedup_mp],
             "Speedup wrt. 1 thread",
             title=graph_label,
             legends=["Multithreading", "Multiprocessing"],
             ideal1=1,
             bounds=dict(ymin=0, xmin=0),
             legendYmax=0.5
    )


if __name__ == "__main__":
    main(sys.argv)
