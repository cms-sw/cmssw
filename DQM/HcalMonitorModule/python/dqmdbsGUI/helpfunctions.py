#!/usr/bin/env python
from Tkinter import *
import os
import string

def Helpwin(filename,usetext=0):

    # If usetext = 1, assume that 'filename' is actually a string
    # containing information to be displayed.
    # Otherwise, assume filename is a file, and open that file
    # to get its contents.
    
    if usetext==1:
        helpfile=filename
    else:
        if os.path.isfile(filename):
            helpfile=open(filename,'r').read()
        else:
            helpfile = "File '%s' does not exist"%filename
    
    try:
        if usetext == 0:
            newwin.title(os.path.basename(filename))
        else:
            newwin.title("DQMfromDBSgui Info")
        text.delete('1.0',END)
        text.insert('1.0',helpfile)
        text.focus()

    except:
        newwin=Toplevel()
        
        
        if string.find(os.path.basename(filename),"params.dat")>-1:
            newwin.geometry('950x500+20+300')
        else:
            newwin.geometry('950x500+20+300')

        if usetext == 0:
            newwin.title(os.path.basename(filename))
        else:
            newwin.title("Help Information")
        Button(newwin, text="Close", command=newwin.destroy).pack()
        ybar=Scrollbar(newwin)
        text=Text(newwin,relief=SUNKEN,bg="grey98")
        ybar.config(command=text.yview)
        text.config(yscrollcommand=ybar.set)
        ybar.pack(side=RIGHT,fill=Y)
        text.pack(side=LEFT,expand=YES,fill=BOTH)
        text.delete('1.0',END)
        text.insert('1.0',helpfile)
        text.focus()
        text.bind('<Button-3>',(lambda event,x=newwin:
                                x.destroy()))


def gracefulexit(win):
    win.window.destroy()
