import ROOT
import sys
import numpy as np

def generate_graph(filename, dataset, title, param_no, db, plane):
    file = ROOT.TFile.Open("visualization/"+filename+".root", "RECREATE")

    arr = np.loadtxt("visualization/"+dataset+".dat")
    rows, columns = arr.shape
    x_line = []
    y_line = []

    for r in range(rows):
        if int(arr[r,0]) == int(param_no) and int(arr[r,1]) == int(db) and int(arr[r,2]) == int(plane):
            x_line.append(arr[r,3])
            y_line.append(arr[r,5])

    size = len(x_line)

    x = ROOT.std.vector('double')()
    for i in x_line: x.push_back(float(i))
    y = ROOT.std.vector('double')()
    for i in y_line: y.push_back(float(i))

    c1 = ROOT.TCanvas()
    gr = ROOT.TGraph(size, x.data(), y.data())
    gr.SetTitle(title)
    gr.SetMarkerColor(4)
    gr.SetMarkerStyle(21)
    #gr.Draw("AL")
    file.WriteObject(gr, title)

if __name__ == '__main__':
    generate_graph(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])