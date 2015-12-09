import os,sys,time,calendar
from datetime import datetime,timedelta

class lumiTime(object):
    def __init__(self):
        self.coraltimefm='MM/DD/YY HH24:MI:SS'
        self.pydatetimefm='%m/%d/%y %H:%M:%S'
        self.nbx=3564
        self.orbits_per_ls=262144 # a lumisection is 2**18 orbits
        self.bunchspace_us=0.02495 #in microseconds
        self.bunchspace_s=24.95e-09 #in seconds
        
    #def timestampTodatetimeUTC(self,ts):
        ##return datetime.fromtimestamp(ts,tz=pytz.utc)
        #return datetime.utcfromtimestamp(ts)
    
    def LSDuration(self):
        return self.OrbitDuration(self.orbits_per_ls)

    def OrbitDuration(self, norbits = 1):
        return timedelta(microseconds=(self.nbx*norbits*self.bunchspace_us))
    
    def OrbitToTimeStr(self,begStrTime,orbitnumber,begorbit=0,customfm=''):
        '''
        given a orbit number, return its corresponding time. Assuming begin time has orbit=0
        '''
        return self.DatetimeToStr(self.StrToDatetime(begStrTime)+self.OrbitDuration(orbitnumber-begorbit),customfm=customfm)

    def OrbitToTime(self,begStrTime,orbitnumber,begorbit=0,customfm=''):
        '''
        given a orbit number, return its corresponding time. Default run begin time counting from orbit=0
        '''
        return self.StrToDatetime(begStrTime,customfm=customfm)+self.OrbitDuration(orbitnumber-begorbit)

    def OrbitToLocalTimestamp(self,begStrTime,orbitnumber,begorbit=0,customfm=''):
        '''
        given a orbit number, return its corresponding unixtimestamp. Default run begin time counting from orbit=0
        '''
        os.environ['TZ']='CET'
        time.tzset()
        orbittime=self.OrbitToTime(begStrTime,orbitnumber,begorbit=begorbit,customfm=customfm)
        return time.mktime(orbittime.timetuple())+orbittime.microsecond/1e6

    def OrbitToUTCTimestamp(self,begStrTime,orbitnumber,begorbit=0,customfm=''):
        '''
        given a orbit number, return its corresponding unixtimestamp. Default run begin time counting from orbit=0
        '''
        os.environ['TZ']='UTC'
        time.tzset()
        orbittime=self.OrbitToTime(begStrTime,orbitnumber,begorbit=begorbit,customfm=customfm)
        return time.mktime(orbittime.timetuple())+(orbittime.microsecond/1e6)
    def StrToDatetime(self,strTime,customfm=''):
        '''convert string timestamp to python datetime
        '''
        result=''
        try:
            if not customfm:
                result=datetime.strptime(strTime,self.pydatetimefm)
            else:
                result=datetime.strptime(strTime,customfm)
        except Exception as er:
            print str(er)
        return result
    def DatetimeToStr(self,timeValue,customfm=''):
        '''convert python datetime to string timestamp
        '''
        result=''
        try:
            if not customfm:
                result=timeValue.strftime(self.pydatetimefm)
            else:
                result=timeValue.strftime(customfm)
        except Exception as er:
            print str(er)
        return result
if __name__=='__main__':
    begTimeStr='03/30/10 10:10:01'
    c=lumiTime()
    print 'orbit 0 : ',c.OrbitToTime(begTimeStr,0,0)
    print 'orbit 1 : ',c.OrbitToTime(begTimeStr,1,0)
    print 'orbit 262144 : ',c.OrbitToTime(begTimeStr,262144,0)
    print 'orbit 0 : ',c.OrbitToUTCTimestamp(begTimeStr,0,0);
    print 'orbit 0 : ',c.OrbitToLocalTimestamp(begTimeStr,0,0);
