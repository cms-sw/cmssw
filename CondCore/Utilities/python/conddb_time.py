import datetime

datetime_string_fmt = '%Y-%m-%d %H:%M:%S.%f'

def to_lumi_time( runNumber, lumiSectionId ):
    return (runNumber<<32) + lumiSectionId

def from_lumi_time( lumiTime ):
    run = lumiTime>>32
    lumisection_id = lumiTime-( run << 32 )
    return run, lumisection_id

def to_timestamp( dt ):
    timespan_from_epoch = dt - datetime.datetime(1970,1,1)
    seconds_from_epoch = int( timespan_from_epoch.total_seconds() )
    nanoseconds_from_epoch = timespan_from_epoch.microseconds * 1000
    return ( seconds_from_epoch << 32 ) + nanoseconds_from_epoch

def from_timestamp( ts ):
    seconds_from_epoch = ts >> 32
    nanoseconds_from_epoch = ts - ( seconds_from_epoch << 32 )
    dt = datetime.datetime.utcfromtimestamp(seconds_from_epoch)
    return dt + datetime.timedelta( microseconds=int(nanoseconds_from_epoch/1000) )

def string_to_timestamp( sdt ):
    dt = datetime.datetime.strptime( sdt, datetime_string_fmt )
    return to_timestamp( dt )

def string_from_timestamp( ts ):
    dt = from_timestamp( ts )
    return dt.strftime( datetime_string_fmt )


