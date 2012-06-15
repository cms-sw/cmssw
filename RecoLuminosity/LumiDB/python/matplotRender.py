'''
Specs:
-- We use matplotlib OO class level api, we do not use its high-level helper modules. Favor endured stability over simplicity. 
-- PNG as default batch file format
-- we support http mode by sending string buf via meme type image/png. Sending a premade static plot to webserver is considered a uploading process instead of http dynamic graphical mode. 
'''
import sys,os
import numpy,datetime
import matplotlib
from RecoLuminosity.LumiDB import CommonUtil,lumiTime,csvReporter

batchonly=False
if not os.environ.has_key('DISPLAY') or not os.environ['DISPLAY']:
    batchonly=True
    matplotlib.use('Agg',warn=False)
else:
    try:
        from RecoLuminosity.LumiDB import lumiQTWidget  
    except ImportError:
        print 'unable to import GUI backend, switch to batch only mode'
        matplotlib.use('Agg',warn=False)
        batchonly=True
from matplotlib.backends.backend_agg import FigureCanvasAgg as CanvasBackend
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager,FontProperties
matplotlib.rcParams['lines.linewidth']=1.5
matplotlib.rcParams['grid.linewidth']=0.2
matplotlib.rcParams['xtick.labelsize']=11
matplotlib.rcParams['ytick.labelsize']=11
matplotlib.rcParams['legend.fontsize']=10
matplotlib.rcParams['axes.labelsize']=11
matplotlib.rcParams['font.weight']=567

def guessLumiUnit(t):
    '''
    input : largest total lumivalue
    output: (unitstring,denomitor)
    '''
    unitstring='$\mu$b$^{-1}$'
    denomitor=1.0
    if t>=1.0e3 and t<1.0e06:
        denomitor=1.0e3
        unitstring='nb$^{-1}$'
    elif t>=1.0e6 and t<1.0e9:
        denomitor=1.0e6
        unitstring='pb$^{-1}$'
    elif t>=1.0e9 and t<1.0e12:
        denomitor=1.0e9
        unitstring='fb$^{-1}$'
    elif  t>=1.0e12 and t<1.0e15:
        denomitor=1.0e12
        unitstring='ab$^{-1}$'
    elif t<=1.0e-3 and t>1.0e-6: #left direction
        denomitor=1.0e-3
        unitstring='mb$^{-1}$'
    elif t<=1.0e-6 and t>1.0e-9:
        denomitor=1.0e-6
        unitstring='b$^{-1}$'
    elif t<=1.0e-9 and t>1.0e-12:
        denomitor=1.0e-9
        unitstring='kb$^{-1}$'
    return (unitstring,denomitor)

class matplotRender():
    def __init__(self,fig):
        self.__fig=fig
        self.__canvas=''
        self.colormap={}
        self.colormap['Delivered']='r'
        self.colormap['Recorded']='b'
        self.colormap['Effective']='g'
        self.colormap['Max Inst']='r'

    def plotSumX_Run(self,rawdata={},resultlines=[],minRun=None,maxRun=None,nticks=6,yscale='linear',withannotation=False,referenceLabel='Delivered',labels=['Delivered','Recorded'],textoutput=None):
        '''
        input:
          rawdata = {'Delivered':[(runnumber,lumiperrun),..],'Recorded':[(runnumber,lumiperrun),..]}
          resultlines = [[runnumber,dellumiperrun,reclumiperrun],[runnumber,dellumiperrun,reclumiperrun],]
          minRun : minimal runnumber required
          maxRun : max runnumber required
          yscale: linear,log or both
          withannotation: wheather the boundary points should be annotated
          referenceLabel: the one variable that decides the total unit and the plot x-axis range
          labels: labels of the variables to plot
          textoutput: text output file name. 
        '''
        ypoints={}
        ytotal={}
        for r in resultlines:#parse old text data
            runnumber=int(r[0])
            if rawdata and runnumber in [t[0] for t in rawdata[referenceLabel]]:continue#use text input only if not in selected data
            if minRun and runnumber<minRun: continue
            if maxRun and runnumber>maxRun: continue
            for i,lab in enumerate(labels) :
                v=float(r[-(len(labels)-i)])#the values to plot are always the last n fields
                rawdata.setdefault(lab,[]).append((runnumber,v))
        if not rawdata:
            print '[WARNING]: no data to plot , exit'
            return
      
        tot=sum([t[1] for t in rawdata[referenceLabel]])
        (unitstring,denomitor)=guessLumiUnit(tot)
        csvreport=None
        rows=[]
        flat=[]
        for label,yvalues in rawdata.items():
            yvalues.sort()
            flat.append([t[1] for t in yvalues])
            ypoints[label]=[]
            ytotal[label]=0.0
            lumivals=[t[1] for t in yvalues]
            for i,val in enumerate(lumivals):
                ypoints[label].append(sum(lumivals[0:i+1])/denomitor)#integrated lumi
            ytotal[label]=sum(lumivals)/denomitor
        xpoints=[t[0] for t in rawdata[referenceLabel]]
        ax=self.__fig.add_subplot(111)
        if yscale=='linear':
            ax.set_yscale('linear')
        elif yscale=='log':
            ax.set_yscale('log')
        else:
            raise 'unsupported yscale ',yscale
        ax.set_xlabel(r'Run',position=(0.95,0))
        ax.set_ylabel(r'L '+unitstring,position=(0,0.9))
        xticklabels=ax.get_xticklabels()
        for tx in xticklabels:
            tx.set_rotation(30)
        majorLocator=matplotlib.ticker.LinearLocator( nticks )
        majorFormatter=matplotlib.ticker.FormatStrFormatter('%d')
        minorLocator=matplotlib.ticker.LinearLocator(numticks=4*nticks)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.set_xbound(lower=xpoints[0],upper=xpoints[-1])
        ax.grid(True)
        keylist=ypoints.keys()
        keylist.sort()
        keylist.insert(0,keylist.pop(keylist.index(referenceLabel)))#move refereceLabel to front from now on
        legendlist=[]
        head=['#Run']
        textsummaryhead=['#TotalRun']
        textsummaryline=['#'+str(len(xpoints))]
        for ylabel in keylist:
            cl='k'
            if self.colormap.has_key(ylabel):
                cl=self.colormap[ylabel]
            ax.plot(xpoints,ypoints[ylabel],label=ylabel,color=cl,drawstyle='steps')
            legendlist.append(ylabel+' '+'%.3f'%(ytotal[ylabel])+' '+unitstring)
            textsummaryhead.append('Total'+ylabel)
            textsummaryline.append('%.3f'%(ytotal[ylabel])+' '+unitstring)
            head.append(ylabel)
        if textoutput:
            csvreport=csvReporter.csvReporter(textoutput)
            csvreport.writeRow(head)
            allruns=[int(t[0]) for t in rawdata[referenceLabel]]
            flat.insert(0,allruns)
            rows=zip(*flat)
            csvreport.writeRows([list(t) for t in rows])
            csvreport.writeRow(textsummaryhead)
            csvreport.writeRow(textsummaryline)
        #font=FontProperties(size='medium',weight='demibold')
        #legend
        ax.legend(tuple(legendlist),loc='upper left')
        #adjust
        self.__fig.subplots_adjust(bottom=0.18,left=0.1)
        #annotations
        if withannotation:
            trans=matplotlib.transforms.BlendedGenericTransform(ax.transData,ax.transAxes)
            ax.text(xpoints[0],1.025,str(xpoints[0]),transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))
            ax.text(xpoints[-1],1.025,str(xpoints[-1]),transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))
        
    
    def plotSumX_Fill(self,rawdata={},resultlines=[],minFill=None,maxFill=None,nticks=6,yscale='linear',withannotation=False,referenceLabel='Delivered',labels=['Delivered','Recorded'],textoutput=None):
        '''
        input:
        rawdata = {'Delivered':[(fill,runnumber,lumiperrun)],'Recorded':[(fill,runnumber,lumiperrun)]}
        resultlines = [[fillnumber,runnumber,dellumiperrun,reclumiperrun],[fillnumber,runnumber,dellumiperrun,reclumiperrun],]
        minFill : min fill to draw
        maxFill : max fill to draw
        yscale: linear,log or both
        withannotation: wheather the boundary points should be annotated
        textoutput: text output file name. 
        '''
        ytotal={}
        ypoints={}
        for r in resultlines: #parse old text data
            fillnum=int(r[0])
            runnum=int(r[1])
            if rawdata and (fillnum,runnum) in [(t[0],t[1]) for t in rawdata[referenceLabel]]:continue
            if minFill and fillnum<minFill:continue
            if maxFill and fillnum>maxFill:continue
            for i,lab in enumerate(labels) :
                v=float(r[-(len(labels)-i)])#the values to plot are always the last n fields
                rawdata.setdefault(lab,[]).append((fillnum,runnum,v))
        #print 'fillrunDict ',fillrunDict
        if not rawdata:
            print '[WARNING]: no data, do nothing'
            return
        tot=sum([t[2] for t in rawdata[referenceLabel]])
        beginfo=''
        endinfo=''
        (unitstring,denomitor)=guessLumiUnit(tot)
        csvreport=None
        rows=[]
        flat=[]
        for label,yvalues in rawdata.items():
            yvalues.sort()
            flat.append([t[2] for t in yvalues])
            ypoints[label]=[]
            ytotal[label]=0.0
            lumivals=[t[2] for t in yvalues]
            for i,val in enumerate(lumivals):
                ypoints[label].append(sum(lumivals[0:i+1])/denomitor)
            ytotal[label]=sum(lumivals)/denomitor
        xpoints=[t[0] for t in rawdata[referenceLabel]]#after sort
        ax=self.__fig.add_subplot(111)
        ax.set_xlabel(r'LHC Fill Number',position=(0.84,0))
        ax.set_ylabel(r'L '+unitstring,position=(0,0.9))
        ax.set_xbound(lower=xpoints[0],upper=xpoints[-1])
        if yscale=='linear':
            ax.set_yscale('linear')
        elif yscale=='log':
            ax.set_yscale('log')
        else:
            raise 'unsupported yscale ',yscale
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
        keylist.insert(0,keylist.pop(keylist.index(referenceLabel)))#move refereceLabel to front from now on
        legendlist=[]
        head=['#fill','run']        
        textsummaryhead=['#TotalFill']
        textsummaryline=['#'+str(len(xpoints))]
        for ylabel in keylist:
            cl='k'
            if self.colormap.has_key(ylabel):
                cl=self.colormap[ylabel]
            ax.plot(xpoints,ypoints[ylabel],label=ylabel,color=cl,drawstyle='steps')
            legendlist.append(ylabel+' '+'%.3f'%(ytotal[ylabel])+' '+unitstring)
            textsummaryhead.append('Total'+ylabel)
            textsummaryline.append('%.3f'%(ytotal[ylabel])+' '+unitstring)
            head.append(ylabel)
        if textoutput:
            csvreport=csvReporter.csvReporter(textoutput)
            allfills=[int(t[0]) for t in rawdata[referenceLabel]]
            allruns=[int(t[1]) for t in rawdata[referenceLabel]]
            flat.insert(0,allfills)
            flat.insert(1,allruns)
            rows=zip(*flat)
            csvreport.writeRow(head)
            csvreport.writeRows([list(t) for t in rows])
            csvreport.writeRow(textsummaryhead)
            csvreport.writeRow(textsummaryline)
        #font=FontProperties(size='medium',weight='demibold')
        #annotations
        if withannotation:
            trans=matplotlib.transforms.BlendedGenericTransform(ax.transData,ax.transAxes)
            ax.text(xpoints[0],1.025,beginfo,transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))
            ax.text(xpoints[-1],1.025,endinfo,transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))
        #legend
        ax.legend(tuple(legendlist),loc='upper left')
        #adjust
        self.__fig.subplots_adjust(bottom=0.1,left=0.1)
        
    def plotSumX_Time(self,rawdata={},resultlines=[],minTime=None,maxTime=None,nticks=6,yscale='linear',withannotation=False,referenceLabel='Delivered',labels=['Delivered','Recorded'],textoutput=None):
        '''
        input:
        rawdata = {'Delivered':[(runnumber,starttimestamp,stoptimestamp,lumiperrun)],'Recorded':[(runnumber,starttimestamp,stoptimestamp,lumiperrun)]}
        resultlines = [[runnumber,starttimestampStr,stoptimestampStr,dellumiperrun,reclumiperrun],[runnumber,starttimestampStr,stoptimestampStr,dellumiperrun,reclumiperrun],]
        minTime (python DateTime) : min *begin* time to draw: format %m/%d/%y %H:%M:%S
        maxTime (python DateTime): max *begin* time to draw %m/%d/%y %H:%M:%S
        yscale: linear,log or both
        withannotation: wheather the boundary points should be annotated
        referenceLabel: the one variable that decides the total unit and the plot x-axis range
        labels: labels of the variables to plot
        '''
        xpoints=[]
        ypoints={}
        ytotal={}
        lut=lumiTime.lumiTime()
        if not minTime:
            minTime='03/01/10 00:00:00'
        minTime=lut.StrToDatetime(minTime,customfm='%m/%d/%y %H:%M:%S')
        if not maxTime:
            maxTime=datetime.datetime.utcnow()
        else:
            maxTime=lut.StrToDatetime(maxTime,customfm='%m/%d/%y %H:%M:%S')
        for r in resultlines:
            runnumber=int(r[0])
            starttimeStr=r[1].split('.')[0]
            starttime=lut.StrToDatetime(starttimeStr,customfm='%Y-%m-%d %H:%M:%S')
            stoptimeStr=r[2].split('.')[0]
            stoptime=lut.StrToDatetime(stoptimeStr,customfm='%Y-%m-%d %H:%M:%S')
            if rawdata and runnumber in [t[0] for t in rawdata[referenceLabel]]:continue
            if starttime<minTime:continue
            if starttime>maxTime:continue
                
            for i,lab in enumerate(labels):
                v=float(r[-(len(labels)-i)])
                rawdata.setdefault(lab,[]).append((runnumber,starttime,stoptime,v))        
        if not rawdata:
            print '[WARNING]: no data, do nothing'
            return
        tot=sum([t[3] for t in rawdata[referenceLabel]])
        (unitstring,denomitor)=guessLumiUnit(tot)
        csvreport=None
        rows=[]
        flat=[]
        for label,yvalues in rawdata.items():
            yvalues.sort()
            flat.append([t[3] for t in yvalues])
            if label==referenceLabel:
                minTime=yvalues[0][1]
                maxTime=yvalues[-1][1]
            ypoints[label]=[]
            lumivals=[t[3] for t in yvalues]
            for i,val in enumerate(lumivals):
                ypoints[label].append(sum(lumivals[0:i+1])/denomitor)
            ytotal[label]=sum(lumivals)/denomitor
        xpoints=[matplotlib.dates.date2num(t[1]) for t in rawdata[referenceLabel]]
        ax=self.__fig.add_subplot(111)
        ax.set_yscale(yscale)
        yearStrMin=minTime.strftime('%Y')
        yearStrMax=maxTime.strftime('%Y')
        if yearStrMin==yearStrMax:
            dateFmt=matplotlib.dates.DateFormatter('%d/%m')
        else:
            dateFmt=matplotlib.dates.DateFormatter('%d/%m/%y')
        majorLoc=matplotlib.ticker.LinearLocator(numticks=nticks)
        ax.xaxis.set_major_locator(majorLoc)
        minorLoc=matplotlib.ticker.LinearLocator(numticks=nticks*4)
        ax.xaxis.set_major_formatter(dateFmt)
        ax.set_xlabel(r'Date',position=(0.84,0))
        ax.set_ylabel(r'L '+unitstring,position=(0,0.9))
        ax.xaxis.set_minor_locator(minorLoc)
        ax.set_xbound(lower=xpoints[0],upper=xpoints[-1])
        xticklabels=ax.get_xticklabels()
        for tx in xticklabels:
            tx.set_horizontalalignment('left')
        ax.grid(True)
        keylist=ypoints.keys()
        keylist.sort()
        keylist.insert(0,keylist.pop(keylist.index(referenceLabel)))#move refereceLabel to front from now on
        legendlist=[]
        head=['#Run','StartTime','StopTime']
        textsummaryhead=['#TotalRun']
        textsummaryline=['#'+str(len(xpoints))]
        for ylabel in keylist:
            cl='k'
            if self.colormap.has_key(ylabel):
                cl=self.colormap[ylabel]
            ax.plot(xpoints,ypoints[ylabel],label=ylabel,color=cl,drawstyle='steps')
            legendlist.append(ylabel+' '+'%.3f'%(ytotal[ylabel])+' '+unitstring)
            textsummaryhead.append('Total'+ylabel)
            textsummaryline.append('%.3f'%(ytotal[ylabel])+' '+unitstring)
            head.append(ylabel)
        if textoutput:
            csvreport=csvReporter.csvReporter(textoutput)
            csvreport.writeRow(head)
            allruns=[int(t[0]) for t in rawdata[referenceLabel]]
            allstarts=[ t[1] for t in rawdata[referenceLabel]]
            allstops=[ t[2] for t in rawdata[referenceLabel]]
            flat.insert(0,allruns)
            flat.insert(1,allstarts)
            flat.insert(2,allstops)
            rows=zip(*flat)
            csvreport.writeRows([list(t) for t in rows])
            csvreport.writeRow(textsummaryhead)
            csvreport.writeRow(textsummaryline)
        #annotations
        trans=matplotlib.transforms.BlendedGenericTransform(ax.transData,ax.transAxes)
        #print 'run boundary ',runs[0],runs[-1]
        #print 'xpoints boundary ',xpoints[0],xpoints[-1]
        #annotation
        if withannotation:
            runs=[t[0] for t in rawdata[referenceLabel]]
            ax.text(xpoints[0],1.025,str(runs[0]),transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))        
            ax.text(xpoints[-1],1.025,str(runs[-1]),transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))
        
        if yearStrMin==yearStrMax:
            firsttimeStr=rawdata[referenceLabel][1][1].strftime('%b %d %H:%M') #time range(start) in the title is the first run beg time 
            lasttimeStr=rawdata[referenceLabel][-1][2].strftime('%b %d %H:%M') #time range(stop) in the tile is the last run stop time
            #firstimeStr=minTime.strftime('%b %d %H:%M')
            #lasttimeStr=maxTime.strftime('%b %d %H:%M')
            #ax.set_title('CMS Total Integrated Luminosity '+yearStrMin+' ('+firstimeStr+' - '+lasttimeStr+' UTC)',size='small',family='fantasy')
            ax.set_title('CMS Total Integrated Luminosity '+yearStrMin+' ('+firsttimeStr+' - '+lasttimeStr+' UTC)',size='small')
        else:
            #ax.set_title('CMS Total Integrated Luminosity '+yearStrMin+'-'+yearStrMax,size='small',family='fantasy')
            ax.set_title('CMS Total Integrated Luminosity '+yearStrMin+'-'+yearStrMax,size='small')
        ax.legend(tuple(legendlist),loc='upper left')
        ax.autoscale_view(tight=True,scalex=True,scaley=False)
        self.__fig.autofmt_xdate(bottom=0.18,rotation=15,ha='right')
        self.__fig.subplots_adjust(bottom=0.2,left=0.15)
        
    def plotPerdayX_Time(self,rawdata={},resultlines=[],minTime=None,maxTime=None,nticks=6,yscale='linear',withannotation=False,referenceLabel='Delivered',labels=['Delivered','Recorded'],textoutput=None):
        '''
        Input:
        rawdata={'Delivered':[(day,begrun:ls,endrun:ls,lumi)],'Recorded':[(dayofyear,begrun:ls,endrun:ls,lumi)]}
        resultlines=[[day,begrun:ls,endrun:ls,deliveredperday,recordedperday],[]]
        minTime (python DateTime) : min *begin* time to draw: format %m/%d/%y %H:%M:%S
        maxTime (python DateTime): max *begin* time to draw %m/%d/%y %H:%M:%S
        withannotation: wheather the boundary points should be annotated
        referenceLabel: the one variable that decides the total unit and the plot x-axis range
        labels: labels of the variables to plot
        '''
        xpoints=[]
        ypoints={}
        ymax={}
        lut=lumiTime.lumiTime()
        if not minTime:
            minTime='03/01/10 00:00:00'
        minTime=lut.StrToDatetime(minTime,customfm='%m/%d/%y %H:%M:%S')
        if not maxTime:
            maxTime=datetime.datetime.utcnow()
        else:
            maxTime=lut.StrToDatetime(maxTime,customfm='%m/%d/%y %H:%M:%S')
        for r in resultlines:
            day=int(r[0])
            begrunls=r[1]
            endrunls=r[2]
            #[begrun,begls]=[int(s) for s in r[1].split(':')]
            if rawdata and day in [t[0] for t in rawdata[referenceLabel]]:continue
            if day < minTime.date().toordinal():continue
            if day > maxTime.date().toordinal():continue
            for i,lab in enumerate(labels):
                v=float(r[-(len(labels)-i)])
                rawdata.setdefault(lab,[]).append((day,begrunls,endrunls,v))
        if not rawdata:
            print '[WARNING]: no data, do nothing'
            return
        maxlum=max([t[3] for t in rawdata[referenceLabel]])
        minlum=min([t[3] for t in rawdata[referenceLabel] if t[3]>0]) #used only for log scale, fin the non-zero bottom
        (unitstring,denomitor)=guessLumiUnit(maxlum)
        csvreport=None
        rows=[]
        flat=[]
        for label,yvalues in rawdata.items():
            yvalues.sort()
            flat.append([t[3] for t in yvalues])
            minday=yvalues[0][0]
            #print 'minday ',minday
            maxday=yvalues[-1][0]
            #print 'maxday ',maxday
            alldays=range(minday,maxday+1)
            #print 'alldays ',alldays
            ypoints[label]=[]
            dayvals=[t[0] for t in yvalues]
            lumivals=[t[3] for t in yvalues]
            #print 'lumivals ',lumivals
            for d in alldays:
                if not d in dayvals:
                    ypoints[label].append(0.0)
                else:
                    thisdaylumi=[t[3] for t in yvalues if t[0]==d][0]
                    if yscale=='log':
                        if thisdaylumi<minlum:
                            thisdaylumi=minlum/denomitor
                        else:
                            thisdaylumi=thisdaylumi/denomitor
                    else:
                         thisdaylumi=thisdaylumi/denomitor
                    ypoints[label].append(thisdaylumi)
                ymax[label]=max(lumivals)/denomitor
        xpoints=alldays
        if textoutput:
            csvreport=csvReporter.csvReporter(textoutput)
            head=['#day','begrunls','endrunls','delivered','recorded']
            csvreport.writeRow(head)
            flat.insert(0,alldays)
            allstarts=[ t[1] for t in rawdata[referenceLabel]]
            allstops=[ t[2] for t in rawdata[referenceLabel]]
            flat.insert(1,allstarts)
            flat.insert(2,allstops)
            rows=zip(*flat)
            csvreport.writeRows([list(t) for t in rows])
        
        yearStrMin=minTime.strftime('%Y')
        yearStrMax=maxTime.strftime('%Y')
        if yearStrMin==yearStrMax:
            dateFmt=matplotlib.dates.DateFormatter('%d/%m')
        else:
            dateFmt=matplotlib.dates.DateFormatter('%d/%m/%y')
        ax=self.__fig.add_subplot(111)     
        if yscale=='linear':
            ax.set_yscale('linear')
        elif yscale=='log':
            ax.set_yscale('log')
        else:
            raise 'unsupported yscale ',yscale        
        majorLoc=matplotlib.ticker.LinearLocator(numticks=nticks)
        minorLoc=matplotlib.ticker.LinearLocator(numticks=nticks*4)
        ax.xaxis.set_major_formatter(dateFmt)
        ax.set_xlabel(r'Date',position=(0.84,0))
        ax.xaxis.set_major_locator(majorLoc)
        ax.xaxis.set_minor_locator(minorLoc)
        xticklabels=ax.get_xticklabels()
        for tx in xticklabels:
            tx.set_horizontalalignment('right')
        ax.grid(True)
        legendlist=[]
        ax.set_ylabel(r'L '+unitstring,position=(0,0.9))
        textsummaryhead=['#TotalDays']
        textsummaryline=['#'+str(len(xpoints))]
        for ylabel in labels:
            cl='k'
            if self.colormap.has_key(ylabel):
                cl=self.colormap[ylabel]
            ax.plot(xpoints,ypoints[ylabel],label=ylabel,color=cl,drawstyle='steps')
            legendlist.append(ylabel+' Max '+'%.3f'%(ymax[ylabel])+' '+unitstring)
            textsummaryhead.append('Max'+ylabel)
            textsummaryline.append('%.3f'%(ymax[ylabel])+' '+unitstring)
        if textoutput:
            csvreport.writeRow(textsummaryhead)
            csvreport.writeRow(textsummaryline)
        ax.legend(tuple(legendlist),loc='upper left')
        ax.set_xbound(lower=matplotlib.dates.date2num(minTime),upper=matplotlib.dates.date2num(maxTime))
        #if withannotation:
        #        begtime=boundaryInfo[0][0]
        #        beginfo=boundaryInfo[0][1]
        #        endtime=boundaryInfo[1][0]
        #        endinfo=boundaryInfo[1][1]
        #        #annotations
        #        trans=matplotlib.transforms.BlendedGenericTransform(ax.transData,ax.transAxes)
        #        ax.text(matplotlib.dates.date2num(begtime),1.025,beginfo,transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))        
        #        ax.text(matplotlib.dates.date2num(endtime),1.025,endinfo,transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))
        
        firstday=datetime.date.fromordinal(rawdata[referenceLabel][0][0])
        lastday=datetime.date.fromordinal(rawdata[referenceLabel][-1][0])
        firstdayStr=firstday.strftime('%Y %b %d')
        lastdayStr=lastday.strftime('%Y %b %d')
        ax.set_title('CMS Integrated Luminosity/Day ('+firstdayStr+' - '+lastdayStr+')',size='small')
        #ax.autoscale(tight=True)
        ax.autoscale_view(tight=True,scalex=True,scaley=False)
        #ax.set_xmargin(0.015)
        self.__fig.autofmt_xdate(bottom=0.18,rotation=15,ha='right')
        self.__fig.subplots_adjust(bottom=0.2,left=0.15)

    def plotPeakPerday_Time(self,rawdata={},resultlines=[],minTime=None,maxTime=None,nticks=6,withannotation=False,yscale='linear',referenceLabel='Delivered',labels=['Delivered'],textoutput=None):
        '''
        THIS PLOT IS DELIVERED ONLY
        Input:
        rawdata={'Delivered':[(day,run,ls,instlumi)]}
        resultlines=[[day,run,ls,maxinstlum],[]]
        minTime (python DateTime) : min *begin* time to draw: format %m/%d/%y %H:%M:%S
        maxTime (python DateTime): max *begin* time to draw %m/%d/%y %H:%M:%S
        withannotation: wheather the boundary points should be annotated
        referenceLabel: the one variable that decides the total unit and the plot x-axis range
        labels: labels of the variables to plot
        '''
        xpoints=[]
        ypoints={}
        legendlist=[]
        maxinfo=''
        ymax={}
        lut=lumiTime.lumiTime()
        if not minTime:
            minTime='03/01/10 00:00:00'
        minTime=lut.StrToDatetime(minTime,customfm='%m/%d/%y %H:%M:%S')
        if not maxTime:
            maxTime=datetime.datetime.utcnow()
        else:
            maxTime=lut.StrToDatetime(maxTime,customfm='%m/%d/%y %H:%M:%S')
        for r in resultlines:
            day=int(r[0])
            runnumber=int(r[1])
            lsnum=int(r[2].split('.')[0])
            if rawdata and day in [int(t[0]) for t in rawdata[referenceLabel]]:continue
            if day < minTime.date().toordinal():continue
            if day > maxTime.date().toordinal():continue
            for i,lab in enumerate(labels):
                v=float(r[-(len(labels)-i)])
                rawdata.setdefault(lab,[]).append((day,runnumber,lsnum,v))
        if not rawdata:
            print '[WARNING]: no data, do nothing'
            return
        maxlum=max([t[3] for t in rawdata[referenceLabel]])
        minlum=min([t[3] for t in rawdata[referenceLabel] if t[3]>0]) #used only for log scale, fin the non-zero bottom
        (unitstring,denomitor)=guessLumiUnit(maxlum)
        csvreport=None
        rows=[]
        flat=[]
        alldays=[]
        for label,yvalues in rawdata.items():
            yvalues.sort()#sort by day
            minday=yvalues[0][0]
            maxday=yvalues[-1][0]
            alldays=range(minday,maxday+1)
            ypoints[label]=[]
            dayvals=[t[0] for t in yvalues]
            lumivals=[t[3] for t in yvalues]
            flat.append(lumivals)
            for d in alldays:
                if not d in dayvals:
                    ypoints[label].append(0.0)
                else:
                    thisdaylumi=[t[3] for t in yvalues if t[0]==d][0]
                    if yscale=='log':
                        if thisdaylumi<minlum:
                            thisdaylumi=minlum/denomitor
                        else:
                            thisdaylumi=thisdaylumi/denomitor
                    else:
                        thisdaylumi=thisdaylumi/denomitor
                    ypoints[label].append(thisdaylumi)
            ymax[label]=max(lumivals)/denomitor
        xpoints=alldays
        if textoutput:
            csvreport=csvReporter.csvReporter(textoutput)
            head=['#day','run','lsnum','maxinstlumi']
            csvreport.writeRow(head)
            flat.insert(0,[t[0] for t in yvalues])
            allruns=[ t[1] for t in rawdata[referenceLabel]]
            allls=[ t[2] for t in rawdata[referenceLabel]]
            flat.insert(1,allruns)
            flat.insert(2,allls)
            rows=zip(*flat)
            csvreport.writeRows([list(t) for t in rows])
            
        yearStrMin=minTime.strftime('%Y')
        yearStrMax=maxTime.strftime('%Y')
        if yearStrMin==yearStrMax:
            dateFmt=matplotlib.dates.DateFormatter('%d/%m')
        else:
            dateFmt=matplotlib.dates.DateFormatter('%d/%m/%y')
        ax=self.__fig.add_subplot(111)
        if yscale=='linear':
            ax.set_yscale('linear')
        elif yscale=='log':
            ax.set_yscale('log')
        else:
            raise 'unsupported yscale ',yscale
        majorLoc=matplotlib.ticker.LinearLocator(numticks=nticks)
        minorLoc=matplotlib.ticker.LinearLocator(numticks=nticks*4)
        ax.xaxis.set_major_formatter(dateFmt)
        ax.set_xlabel(r'Date',position=(0.84,0))
        ax.set_ylabel(r'L '+unitstring,position=(0,0.9))
        ax.xaxis.set_major_locator(majorLoc)
        ax.xaxis.set_minor_locator(minorLoc)
        xticklabels=ax.get_xticklabels()
        for tx in xticklabels:
            tx.set_horizontalalignment('right')
        ax.grid(True)
        cl=self.colormap['Max Inst']
        textsummaryhead=['#TotalDays']
        textsummaryline=['#'+str(len(xpoints))]
        for ylabel in labels:
            cl='k'
            if self.colormap.has_key(ylabel):
                cl=self.colormap[ylabel]
            ax.plot(xpoints,ypoints[ylabel],label='Max Inst',color=cl,drawstyle='steps')
            legendlist.append('Max Inst %.3f'%(ymax[ylabel])+' '+unitstring)
            textsummaryhead.append('Max Inst'+ylabel)
            textsummaryline.append('%.3f'%(ymax[ylabel])+' '+unitstring)
        if textoutput:
            csvreport.writeRow(textsummaryhead)
            csvreport.writeRow(textsummaryline)
        ax.legend(tuple(legendlist),loc='upper left')
        ax.set_xbound(lower=matplotlib.dates.date2num(minTime),upper=matplotlib.dates.date2num(maxTime))
        if withannotation:
           #annotations
           trans=matplotlib.transforms.BlendedGenericTransform(ax.transData,ax.transAxes)
           ax.text(xpoints[0],1.025,beginfo,transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))
           ax.text(xpoints[-1],1.025,endinfo,transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))
           ax.annotate(maxinfo,xy=(xmax,ymax),xycoords='data',xytext=(0,13),textcoords='offset points',arrowprops=dict(facecolor='green',shrink=0.05),size='x-small',horizontalalignment='center',color='green',bbox=dict(facecolor='white'))
           
        firstday=datetime.date.fromordinal(rawdata[referenceLabel][0][0])
        lastday=datetime.date.fromordinal(rawdata[referenceLabel][-1][0])
        firstdayStr=firstday.strftime('%Y %b %d')
        lastdayStr=lastday.strftime('%Y %b %d')
        ax.set_title('CMS Peak Luminosity/Day ('+firstdayStr+' - '+lastdayStr+')',size='small')

        #ax.autoscale(tight=True)
        ax.autoscale_view(tight=True,scalex=True,scaley=False)        
        #ax.set_xmargin(0.015)
        self.__fig.autofmt_xdate(bottom=0.18,rotation=15,ha='right')
        self.__fig.subplots_adjust(bottom=0.2,left=0.15)

    def plotInst_RunLS(self,rawxdata,rawydata,nticks=6,textoutput=None):
        '''
        Input: rawxdata [run,fill,starttime,stoptime,totalls,ncmsls]
               rawydata {label:[lumi]}
        '''
        lslength=23.357
        lut=lumiTime.lumiTime()
        runnum=rawxdata[0]
        fill=rawxdata[1]
        starttime=lut.DatetimeToStr(rawxdata[2],customfm='%m/%d/%y %H:%M:%S')
        stoptime=lut.DatetimeToStr(rawxdata[3],customfm='%m/%d/%y %H:%M:%S')
        totalls=rawxdata[-2]
        ncmsls=rawxdata[-1]
        peakinst=max(rawydata['Delivered'])/lslength
        totaldelivered=sum(rawydata['Delivered'])
        totalrecorded=sum(rawydata['Recorded'])
        xpoints=range(1,totalls+1)        
        #print len(xpoints)
        ypoints={}
        ymax={}
        for ylabel,yvalue in rawydata.items():
            ypoints[ylabel]=[y/lslength for y in yvalue]
            ymax[ylabel]=max(yvalue)/lslength
        left=0.15
        width=0.7
        bottom=0.1
        height=0.65
        bottom_h=bottom+height
        rect_scatter=[left,bottom,width,height]
        rect_table=[left,bottom_h,width,0.25]
        
        nullfmt=matplotlib.ticker.NullFormatter()
        nullloc=matplotlib.ticker.NullLocator()
        axtab=self.__fig.add_axes(rect_table,frameon=False)
        axtab.set_axis_off()
        axtab.xaxis.set_major_formatter(nullfmt)
        axtab.yaxis.set_major_formatter(nullfmt)
        axtab.xaxis.set_major_locator(nullloc)
        axtab.yaxis.set_major_locator(nullloc)

        ax=self.__fig.add_axes(rect_scatter)
        
        majorLoc=matplotlib.ticker.LinearLocator(numticks=nticks)
        minorLoc=matplotlib.ticker.LinearLocator(numticks=nticks*4)
        ax.set_xlabel(r'LS',position=(0.96,0))
        ax.set_ylabel(r'L $\mu$b$^{-1}$s$^{-1}$',position=(0,0.9))
        ax.xaxis.set_major_locator(majorLoc)
        ax.xaxis.set_minor_locator(minorLoc)
        ax.set_xbound(lower=xpoints[0],upper=xpoints[-1])
        xticklabels=ax.get_xticklabels()
        for tx in xticklabels:
            tx.set_horizontalalignment('right')
        ax.grid(True)
        keylist=ypoints.keys()
        keylist.sort()
        legendlist=[]

        for ylabel in keylist:
            cl='k'
            if self.colormap.has_key(ylabel):
                cl=self.colormap[ylabel]
            ax.plot(xpoints,ypoints[ylabel],'.',label=ylabel,color=cl)
            legendlist.append(ylabel)      
        #ax.axhline(0,color='green',linewidth=0.2)
        ax.axvline(xpoints[ncmsls-1],color='green',linewidth=0.2)
        (unitstring,denomitor)=guessLumiUnit(totaldelivered)
        colLabels=('run','fill','max inst(/$\mu$b/s)','delivered('+unitstring+')','recorded('+unitstring+')')
        cellText=[[str(runnum),str(fill),'%.3f'%(peakinst),'%.3f'%(totaldelivered/denomitor),'%.3f'%(totalrecorded/denomitor)]]
       
        sumtable=axtab.table(cellText=cellText,colLabels=colLabels,colWidths=[0.12,0.1,0.27,0.27,0.27],cellLoc='center',loc='center')
        trans=matplotlib.transforms.BlendedGenericTransform(ax.transData,ax.transAxes)        
        axtab.add_table(sumtable)
        
        ax.text(xpoints[0],1.02,starttime[0:17],transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))   
        ax.text(xpoints[ncmsls-1],1.02,stoptime[0:17],transform=trans,horizontalalignment='left',size='x-small',color='green',bbox=dict(facecolor='white'))        
        ax.legend(tuple(legendlist),loc='upper right',numpoints=1)

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
        aw=lumiQTWidget.ApplicationWindow(fig=self.__fig)
        aw.show()
        aw.destroy()
        
if __name__=='__main__':
    import csv
    print '=====testing plotSumX_Run======'
    f=open('/afs/cern.ch/cms/lumi/www/plots/operation/totallumivsrun-2011.csv','r')
    reader=csv.reader(f,delimiter=',')
    resultlines=[]
    for row in reader:
        if not row[0].isdigit():continue
        resultlines.append(row)
    #print resultlines
    fig=Figure(figsize=(7.2,5.4),dpi=120)
    m=matplotRender(fig)
    m.plotSumX_Run(rawdata={},resultlines=resultlines,minRun=None,maxRun=None,nticks=6,yscale='linear',withannotation=False)
    #m.drawPNG('totallumivsrun-2011test.png')
    m.drawInteractive()
    print 'DONE'
    
'''
    print '=====testing plotSumX_Fill======'
    f=open('/afs/cern.ch/cms/lumi/www/plots/operation/totallumivsfill-2011.csv','r')
    reader=csv.reader(f,delimiter=',')
    resultlines=[]
    for row in reader:
        if not row[0].isdigit():continue
        resultlines.append(row)
    #print resultlines
    fig=Figure(figsize=(7.2,5.4),dpi=120)
    m=matplotRender(fig)
    m.plotSumX_Fill(rawdata={},resultlines=resultlines,minFill=None,maxFill=None,nticks=6,yscale='linear',withannotation=True)
    m.drawPNG('totallumivsfill-2011test.png')
    print 'DONE'
    print '=====testing plotSumX_Time======'
    f=open('/afs/cern.ch/cms/lumi/www/publicplots/totallumivstime-2011.csv','r')
    reader=csv.reader(f,delimiter=',')
    resultlines=[]
    for row in reader:
        if not row[0].isdigit():continue
        resultlines.append(row)
    #print resultlines
    fig=Figure(figsize=(7.25,5.4),dpi=120)
    m=matplotRender(fig)
    m.plotSumX_Time(rawdata={},resultlines=resultlines,minTime="03/14/11 09:00:00",maxTime=None,nticks=6,yscale='linear',withannotation=False)
    m.drawPNG('totallumivstime-2011test.png')
    print 'DONE'
    
    print '=====testing plotPerdayX_Time======'
    f=open('/afs/cern.ch/cms/lumi/www/publicplots/lumiperday-2011.csv','r')
    reader=csv.reader(f,delimiter=',')
    resultlines=[]
    for row in reader:
        if not row[0].isdigit():continue
        resultlines.append(row)
    #print resultlines
    fig=Figure(figsize=(7.25,5.4),dpi=120)
    m=matplotRender(fig)
    m.plotPerdayX_Time(rawdata={},resultlines=resultlines,minTime="03/14/11 09:00:00",maxTime=None,nticks=6,yscale='linear',withannotation=False)
    m.drawPNG('lumiperday-2011test.png')
    print 'DONE'

    print '=====testing plotPeakPerday_Time======'
    f=open('/afs/cern.ch/cms/lumi/www/publicplots/lumipeak-2011.csv','r')
    reader=csv.reader(f,delimiter=',')
    resultlines=[]
    for row in reader:
        if not row[0].isdigit():continue
        resultlines.append(row)
    #print resultlines
    fig=Figure(figsize=(7.25,5.4),dpi=120)
    m=matplotRender(fig)
    m.plotPeakPerday_Time(rawdata={},resultlines=resultlines,minTime="03/14/11 09:00:00",maxTime=None,nticks=6,yscale='linear',withannotation=False)
    m.drawPNG('lumipeak-2011test.png')
    print 'DONE'
    
'''
