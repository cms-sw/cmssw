import sys
from numpy import arange,sin,pi,random

batchonly=False
def destroy(e) :
    sys.exit()
import matplotlib
try:
    matplotlib.use('TkAgg',warn=False)
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as CanvasBackend
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
    import Tkinter as Tk
    root=Tk.Tk()
    root.wm_title("Embedding in TK")
except ImportError:
    print 'unable to import GUI backend, switch to batch only mode'
    matplotlib.use('Agg',warn=False)
    from matplotlib.backends.backend_agg import FigureCanvasAgg as CanvasBackend
    batchonly=True

from matplotlib.figure import Figure
import matplotlib.ticker as ticker
 
def drawHTTPstring(fig):
    canvas=CanvasBackend(fig)    
    cherrypy.response.headers['Content-Type']='image/png'
    buf=StringIO()
    canvas.print_png(buf)
    return buf.getvalue()
    
def drawBatch(fig,filename):
    canvas=CanvasBackend(fig)    
    canvas.print_figure(filename)
    
def drawInteractive(fig):
    if batchonly:
        print 'interactive mode is not available for your setup, exit'
        sys.exit()    
    canvas=CanvasBackend(fig,master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)
    toolbar=NavigationToolbar2TkAgg(canvas,root)
    toolbar.update()
    canvas._tkcanvas.pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)
    button = Tk.Button(master=root,text='Quit',command=sys.exit)
    button.pack(side=Tk.BOTTOM)
    Tk.mainloop()

def plotDate(fig):
    import datetime as dt
    ax2=fig.add_subplot(111)
    date2_1=dt.datetime(2008,9,23)
    date2_2=dt.datetime(2008,10,3)
    delta2=dt.timedelta(days=1)
    dates2=matplotlib.dates.drange(date2_1,date2_2,delta2)
    y2=random.rand(len(dates2))
    ax2.set_ylabel(r'Luminosity $\mu$b$^{-1}$')
    ax2.plot_date(dates2,y2,linestyle='-')
    dateFmt=matplotlib.dates.DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_formatter(dateFmt)
    daysLoc=matplotlib.dates.DayLocator()
    hoursLoc=matplotlib.dates.HourLocator(interval=6)
    ax2.xaxis.set_major_locator(daysLoc)
    ax2.xaxis.set_minor_locator(hoursLoc)
    fig.autofmt_xdate(bottom=0.18)
    fig.subplots_adjust(left=0.18)

def plotRun(fig):
    ax=fig.add_subplot(111)
    ax.set_xlabel(r'Run')
    ax.set_ylabel(r'Luminosity $\mu$b$^{-1}$')
    runlist=[136088,136089,136889,136960,137892]
    lumivalues=[0.3,0.6,0.7,0.8,1.0]
    #ax.set_xticklabels(runlist)
    xticklabels=ax.get_xticklabels()
    for tx in xticklabels:
        tx.set_rotation(30)
    minorLocator=matplotlib.ticker.MultipleLocator(100)
    ax.xaxis.set_minor_locator(minorLocator)
    #ax.xaxis.set_major_locator(matplotlib.ticker.LinearLocator(7)
    ax.plot(runlist,lumivalues)
    ax.plot(runlist,[0.8*x for x in lumivalues])
    ax.grid(True)
    fig.subplots_adjust(bottom=0.18,left=0.18)
def plotHist(fig):
    x=[1,2,3,4,5,6]
    y=[1,2,3,4,5,6]
    binsize=1
    ax=fig.add_subplot(111)
    ax.set_xlabel(r'Run')
    ax.set_ylabel(r'Luminosity $\mu$b$^{-1}$')
    print binsize
    #ax.bar(x,y,width=binsize,drawstyle='steps',edgecolor='r',fill=False,label='Recorded')
    ax.plot(x,y,drawstyle='steps')
    ax.grid(True)
    ax.legend()
    fig.subplots_adjust(bottom=0.18,left=0.18)
if __name__=='__main__':
    fig=Figure(figsize=(5,4),dpi=100)
    #a=fig.add_subplot(111)
    #timevars=[1,2,3,4] #should be a absolute packed number runnumber+lsnumber
    #lumivars=[5,6,7,8]
    #use major and minor tickers: major is run,fill or time interval, minor ticker is lumisection. grid is set on major ticker
    #a.set_title('luminosity run')
    #a.set_xlabel('lumi section')
    #a.set_ylabel('Luminosity')
    #a.set_xbound(lower=0,upper=5)
    #a.set_ybound(lower=0.0,upper=10.5)
    #a.set_xticks(range(0,5))
    #a.set_xticks(range(1,5,1))
    #a.plot(timevars,lumivars,'rs-',linewidth=1.0,label='delivered')
    #a.plot(timevars,[v*0.8 for v in lumivars],'gs-',linewidth=1.0,label='recorded')
    #a.grid(True)
    #a.legend(('delivered','recorded'),loc='upper left')
    
    #drawBatch(fig,'testbatch.png')
    #plotDate(fig)
    #plotRun(fig)
    plotHist(fig)
    drawInteractive(fig)
    #print drawHTTPstring()
