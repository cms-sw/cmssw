#!/usr/bin/env perl

opendir THISDIR, "." or die "Whoa! Current directory cannot be opened..";
@allfiles = grep /\.gif$/i, readdir THISDIR;
closedir THISDIR;

foreach $file (@allfiles) {
  $file =~ m/(.+)\.gif$/;
  my $newfilename = $1 . "_small.gif";
  my $command = "convert -geometry 320x $file $newfilename";
  print "$command\n";
  `$command`;
}
