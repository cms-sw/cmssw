import sys
from numpy import arange,sin,pi

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
if __name__=='__main__':
    fig=Figure(figsize=(5,4),dpi=100)
    a=fig.add_subplot(111)
    t=arange(0.0,3.0,0.01)
    s=sin(2*pi*t)
    a.plot(t,s)
    drawBatch(fig,'testbatch.png')
    drawInteractive(fig)
    #print drawHTTPstring()
