#!/usr/bin/env perl

use warnings;
use strict;

use POSIX;

package Util;

sub beginning_time {
  return POSIX::mktime(0,0,0,1,0,105); # 2005-01-01 00:00:00
}

sub to_date {
  my $time = shift;
  return POSIX::strftime("%Y-%m-%d %H:%M:%S", gmtime($time)); 
}

1;
