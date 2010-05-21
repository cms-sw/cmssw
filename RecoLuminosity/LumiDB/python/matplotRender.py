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

def destroy(e) :
    sys.exit()
class matplotRender():
    def __init__(self,fig):
        self.__fig=fig
        self.__canvas=''
    
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
