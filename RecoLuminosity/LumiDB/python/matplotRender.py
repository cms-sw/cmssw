'''
This module is graphical API using pymatplotlib.
Specs:
-- We use matplotlib OO class level api, we do not use its high-level helper modules. Favor endured stability over simplicity. 
-- use TkAgg for interactive mode. Beaware of Tk,pyTk installation defects in various cern distributions.
-- PNG as default batch file format
-- we support http mode by sending string buf via meme type image/png. Sending a premade static plot to webserver is considered a uploading process instead of http dynamic graphical mode. Therefore covered in this module.
'''
import sys
import numpy,datetime
import matplotlib
from RecoLuminosity.LumiDB import CommonUtil

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
matplotlib.rcParams['lines.linewidth']=1.3
matplotlib.rcParams['grid.linewidth']=0.2
matplotlib.rcParams['xtick.labelsize']='medium'
matplotlib.rcParams['ytick.labelsize']='medium'
matplotlib.rcParams['font.weight']='demibold'
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
        self.colormap={}
        self.colormap['Delivered']='r'
        self.colormap['Recorded']='b'
        self.colormap['Effective']='g'

    def plotSumX_Run(self,rawxdata,rawydata,sampleinterval=2,nticks=6):
        xpoints=[]
        ypoints={}
        ytotal={}
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
            ytotal[ylabel]=sum(yvalues)/1000.0    
        ax=self.__fig.add_subplot(111)
        ax.set_xlabel(r'Run',position=(0.95,0))
        ax.set_ylabel(r'L nb$^{-1}$',position=(0,0.9))
        xticklabels=ax.get_xticklabels()
        for tx in xticklabels:
            tx.set_rotation(30)
        majorLocator=matplotlib.ticker.LinearLocator( nticks )
        majorFormatter=matplotlib.ticker.FormatStrFormatter('%d')
        minorLocator=matplotlib.ticker.LinearLocator(numticks=6)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.grid(True)
        keylist=ypoints.keys()
        keylist.sort()
        legendlist=[]
        for ylabel in keylist:
            cl='k'
            if self.colormap.has_key(ylabel):
                cl=self.colormap[ylabel]
            ax.plot(xpoints,ypoints[ylabel],label=ylabel,color=cl,drawstyle='steps')
            legendlist.append(ylabel+' '+'%.2f'%(ytotal[ylabel])+' '+'nb$^{-1}$')
        #font=FontProperties(size='medium',weight='demibold')

        ax.legend(tuple(legendlist),loc='best')
        self.__fig.subplots_adjust(bottom=0.18,left=0.18)
        
    def plotSumX_Fill(self,rawxdata,rawydata,rawfillDict,sampleinterval=2,nticks=6):
        fillboundaries=[]
        xpoints=[]
        ypoints={}
        ytotal={}
        for ylabel in rawydata.keys():
            ypoints[ylabel]=[]
        xidx=[]
        
        #print 'ypoints : ',ypoints
        for ylabel,yvalue in rawydata.items():
            ytotal[ylabel]=sum(rawydata[ylabel])/1000.0
        ax=self.__fig.add_subplot(111)
        ax.set_xlabel(r'LHC Fill Number',position=(0.84,0))
        ax.set_ylabel(r'L nb$^{-1}$',position=(0,0.9))
        xticklabels=ax.get_xticklabels()
        majorLocator=matplotlib.ticker.LinearLocator( nticks )
        majorFormatter=matplotlib.ticker.FormatStrFormatter('%d')
        #minorLocator=matplotlib.ticker.MultipleLocator(sampleinterval)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        #ax.xaxis.set_minor_locator(minorLocator)
        ax.grid(True)
        keylist=ypoints.keys()
        keylist.sort()
        legendlist=[]
        for ylabel in keylist:
            cl='k'
            if self.colormap.has_key(ylabel):
                cl=self.colormap[ylabel]
            ax.plot(xpoints,ypoints[ylabel],label=ylabel,color=cl,drawstyle='steps')
            legendlist.append(ylabel+' '+'%.2f'%(ytotal[ylabel])+' '+'nb$^{-1}$')
        #font=FontProperties(size='medium',weight='demibold')
        ax.legend(tuple(legendlist),loc='best')
        self.__fig.subplots_adjust(bottom=0.18,left=0.3)
        
    def plotSumX_Time(self,rawxdata,rawydata,minTime,maxTime,timedeltahours=3):
        xpoints=[]
        ypoints={}
        ytotal={}
        xidx=[]
        runs=rawxdata.keys()
        runs.sort()
        delta=datetime.timedelta(hours=timedeltahours)
        timesamplers=matplotlib.dates.drange(minTime,maxTime,delta)
        #print 'minTime ordinal ',minTime.toordinal()
        #print 'maxTime ordinal ',maxTime.toordinal()
        #binning in time
        for lowtime,hightime in CommonUtil.pairwise(timesamplers):
            if not hightime: break
            #print 'timesample ',lowtime,hightime
            runsinslot=[]
            for run in runs:
                starttime=rawxdata[run][0]
                stoptime=rawxdata[run][1]
                if starttime.toordinal()>=lowtime and stoptime.toordinal()<hightime and stoptime.toordinal()<=maxTime.toordinal():
                    runsinslot.append(run)
            if len(runsinslot)==0:
                continue
            else:
                binrepsrun=max(runsinslot)
                #print 'max run in bin ',binrepsrun
                xpoints.append(rawxdata[binrepsrun][0].toordinal())
                xidx.append(runs.index(binrepsrun))
        for ylabel,yvalue in rawydata.items():
            ypoints[ylabel]=[]
            for i in xidx:
                ypoints[ylabel].append(sum(yvalue[0:i])/1000.0)
            ytotal[ylabel]=sum(yvalue)/1000.0
        #print 'minTime ',minTime.toordinal()
        #print 'maxTime ',maxTime.toordinal()
        #print 'xpoints ',xpoints
        #print 'ypoints ',ypoints
        ax=self.__fig.add_subplot(111)
        dateFmt=matplotlib.dates.DateFormatter('%d-%m')
        ax.xaxis.set_major_formatter(dateFmt)
        ax.set_xlabel(r'Date',position=(0.84,0))
        ax.set_ylabel(r'L nb$^{-1}$',position=(0,0.9))
        daysLoc=matplotlib.dates.DayLocator(interval=3)
        hoursLoc=matplotlib.dates.HourLocator(interval=12)
        ax.xaxis.set_major_locator(daysLoc)
        ax.xaxis.set_minor_locator(hoursLoc)
        xticklabels=ax.get_xticklabels()
        #for tx in xticklabels:
        #    tx.set_rotation(30)
        ax.grid(True)
        keylist=ypoints.keys()
        keylist.sort()
        legendlist=[]
        for ylabel in keylist:
            cl='k'
            if self.colormap.has_key(ylabel):
                cl=self.colormap[ylabel]
            ax.plot(xpoints,ypoints[ylabel],label=ylabel,color=cl,drawstyle='steps')
            legendlist.append(ylabel+' '+'%.2f'%(ytotal[ylabel])+' '+'nb$^{-1}$')
        #font=FontProperties(size='medium',weight='demibold')
        ax.legend(tuple(legendlist),loc='best')
        self.__fig.autofmt_xdate(bottom=0.18)
        self.__fig.subplots_adjust(bottom=0.18,left=0.3)
        
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
