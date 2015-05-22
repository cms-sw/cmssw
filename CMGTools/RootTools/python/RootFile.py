import ROOT as rt
import os.path

class RootFile(object):
    
    def __init__(self, fileName):
        self.fileName = fileName
        self.plots = {}
        
    def add(self, plot, name = None, dir = None):
        if name is None: name = plot.GetName()
        if dir is not None: name = os.path.join(dir,name)

        l = self.plots.get(name,[])
        l.append(plot)
        self.plots[name] = l

    def write(self):
        out = None
        try:
            out = rt.TFile.Open(self.fileName,'RECREATE')
            for name, plots in self.plots.iteritems():
                out.cd()
                
                dir = os.path.dirname(name)
                objname = os.path.basename(name)
                
                #support for directories if required
                if dir:
                    d = out.Get(dir)
                    if d is not None and d and d.InheritsFrom('TDirectory'):
                        pass
                    else:
                        out.mkdir(dir)
                    out.cd(dir)
                
                if not plots:
                    continue
                elif len(plots) == 1:
                    plots[0].Write(objname)
                else:
                    index = 0
                    for i in xrange(len(plots)):
                        p = plots[i]
                        p.Write('%s_%i' % (objname,i))
            #needed so that the object can be deleted 
            self.plots.clear()
        finally:
            if out is not None: out.Close()
                    