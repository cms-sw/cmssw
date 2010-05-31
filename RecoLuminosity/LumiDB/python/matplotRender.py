import sys
import numpy
import matplotlib
batchonly=False

try:
    matplotlib.use('TkAgg',warn=False)
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as CanvasBackend
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
    import Tkinter as Tk
    root=Tk.Tk()
    root.wm_title("Lumi GUI in TK")
except ImportError:
    print 'unable to import GUI backend, switch to batch only mode'
    matplotlib.use('Agg',warn=False)
    from matplotlib.backends.backend_agg import FigureCanvasAgg as CanvasBackend
    batchonly=True

from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager,FontProperties

def myinclusiveRange(start,stop,step):
    v=start
    while v<stop:
        yield v
        v+=step
    if v>=stop:
        yield stop

def destroy(e) :
    sys.exit()
    
class matplotRender():
    def __init__(self,fig):
        self.__fig=fig
        self.__canvas=''
        
    def plotSumX_Run(self,rawxdata,rawydata,sampleinterval=5,nticks=10):
        xpoints=[]
        ypoints={}
        xidx=[]
        #print 'max rawxdata ',max(rawxdata)
        #print 'min rawxdata ',min(rawxdata)
        for x in myinclusiveRange(min(rawxdata),max(rawxdata),sampleinterval):
            #print 'x : ',x
            xpoints.append(x)
            xidx.append(rawxdata.index(x)) #get the index of the sample points
            #print 'xidx : ',rawxdata.index(x)
        for ylabel,yvalues in rawydata.items():
            ypoints[ylabel]=[]
            for i in xidx:
                ypoints[ylabel].append(sum(yvalues[0:i])/1000.0)                
        ax=self.__fig.add_subplot(111)
        ax.set_xlabel(r'Run')
        ax.set_ylabel(r'L nb$^{-1}$')
        xticklabels=ax.get_xticklabels()
        for tx in xticklabels:
            tx.set_rotation(30)
        #majorLocator=matplotlib.ticker.MultipleLocator( nticks )
        majorLocator=matplotlib.ticker.LinearLocator( nticks )
        majorFormatter=matplotlib.ticker.FormatStrFormatter('%d')
        #minorLocator=matplotlib.ticker.MultipleLocator(5)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        #ax.xaxis.set_minor_locator(minorLocator)
        ax.grid(False)
        for ylabel,yvalue in ypoints.items():
            #print 'plotting x ',xpoints
            #print 'plotting y ',yvalue
            ax.plot(xpoints,yvalue,label=ylabel)

        font=FontProperties(size='small')
        ax.legend(tuple(ypoints.keys()),loc='best',prop=font)

        #legtxt=leg.get_texts()
        #legtxt.size='xx-small'
        self.__fig.subplots_adjust(bottom=0.18,left=0.18)
        
    def plotSumX_Fill(self,rawxdata,rawydata,rawfillDict,sampleinterval=2,nticks=5):
        #rawxdata,rawydata must be equal size
        #calculate tick values
        print 'rawxdata : ',rawxdata
        print 'total : ',len(rawxdata)
        print 'rawydata : ',rawydata
        print 'total delivered : ',len(rawydata.values()[0])
        print 'total recorded : ',len(rawydata.values()[1])
        print 'rawfillDict : ',rawfillDict
        fillboundaries=[]
        xpoints=[]
        ypoints={}
        for ylabel in rawydata.keys():
            ypoints[ylabel]=[]
        xidx=[]
        for x in myinclusiveRange(min(rawfillDict.keys()),max(rawfillDict.keys()),sampleinterval):
            if rawfillDict.has_key(x):
                xpoints.append(x)
        print 'xpoints',xpoints
        
        for fillboundary in xpoints:
            keylist=rawfillDict.keys()
            keylist.sort()
            for fill in keylist:
                if fill==fillboundary:
                    runlist=rawfillDict[fill]
                    runlist.sort()
                    xidx=rawxdata.index(max(runlist))
                    #break
            print 'max runnum for fillboundary ',fillboundary, rawxdata[xidx]
            for ylabel in ypoints:
                ypoints[ylabel].append(sum(rawydata[ylabel][0:xidx])/1000.0)        
        print 'ypoints : ',ypoints            
        ax=self.__fig.add_subplot(111)
        ax.set_xlabel(r'Fill')
        ax.set_ylabel(r'L nb$^{-1}$')
        xticklabels=ax.get_xticklabels()
        majorLocator=matplotlib.ticker.LinearLocator( nticks )
        majorFormatter=matplotlib.ticker.FormatStrFormatter('%d')
        #minorLocator=matplotlib.ticker.MultipleLocator(sampleinterval)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        #ax.xaxis.set_minor_locator(minorLocator)
        ax.grid(True)
        for ylabel,yvalue in ypoints.items():
            ax.plot(xpoints,yvalue,label=ylabel)
        font=FontProperties(size='small')
        ax.legend(tuple(ypoints.keys()),loc='best',prop=font)
        self.__fig.subplots_adjust(bottom=0.18,left=0.18)
        
    def drawHTTPstring(self):
        self.__canvas=CanvasBackend(self.__fig)    
        cherrypy.response.headers['Content-Type']='image/png'
        buf=StringIO()
        self.__canvas.print_png(buf)
        return buf.getvalue()
    
    def drawPNG(self,filename):
        self.__canvas=CanvasBackend(self.__fig)    
        self.__canvas.print_figure(filename)
    
    def drawInteractive(self):
        if batchonly:
            print 'interactive mode is not available for your setup, exit'
            sys.exit()    
        self.__canvas=CanvasBackend(self.__fig,master=root)
        self.__canvas.show()
        self.__canvas.get_tk_widget().pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)
        toolbar=NavigationToolbar2TkAgg(self.__canvas,root)
        toolbar.update()
        self.__canvas._tkcanvas.pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)
        button = Tk.Button(master=root,text='Quit',command=sys.exit)
        button.pack(side=Tk.BOTTOM)
        Tk.mainloop()
if __name__=='__main__':
    fig=Figure(figsize=(5,4),dpi=100)
    a=fig.add_subplot(111)
    t=numpy.arange(0.0,3.0,0.01)
    s=numpy.sin(2*numpy.pi*t)
    a.plot(t,s)
    m=matplotRender(fig)
    m.drawPNG('testmatplotrender.png')
    m.drawInteractive()
    #print drawHTTPstring()   
