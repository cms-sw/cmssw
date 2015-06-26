#!/usr/bin/perl -w

use strict;


my $millescript = "$ENV{'CMSSW_BASE'}/src/Alignment/MillePedeAlignmentAlgorithm/scripts/mps_runMille_template.sh";

my $class = 'cmscaf1nh:cmscafspec1nw';
my $jobname = 'CRUZET2015';
my $pedeScript = "$ENV{'CMSSW_BASE'}/src/Alignment/MillePedeAlignmentAlgorithm/scripts/mps_runPede_rfcp_template.sh";


my $a_pwd = `pwd`;
chomp $a_pwd;

my $mpsdirname = "";

if( my ($path) = $a_pwd =~ /MPproduction\/mp(.+?)$/)
{
    $mpsdirname = "mp$path";
}
else
{
    print "there seems to be a problem to determine the current directory name: $a_pwd\n";
    exit(-1);
}


my $mssDir = "/store/caf/user/$ENV{'USER'}/MPproduction/${mpsdirname}";
print "$mssDir\n";

my $mem = 32768; #16384; #8192;
my $LA = 1;

if(defined $ARGV[0])
{
    $LA = 1 if($ARGV[0] =~ /la/i);
}

my $home = $ENV{'HOME'};

my $mpsdir = "$ENV{'CMSSW_BASE'}/src/Alignment/MillePedeAlignmentAlgorithm/scripts";

my $append = 0;

my $datasetdir = '/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/datasetfiles';
my $jsondir = '/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/datasetfiles';

my $promptglobaltag = 'GR_P_V49::All';

my %confhash = (
    
    'Cosmics0T_cruzet15_6Mar2015_full_stats_part1' => {
        'py' => "$ENV{'PWD'}/alignment_cosmics_0T.py",
        'data' => "${datasetdir}/Cosmics2015/CRUZETandCRAFT2015/Cosmics_233872-236356_CRUZET_Commissioning2015-TkAlCosmics0T-6Mar2015.txt",
        'json' => "${jsondir}/Cosmics2015/CRUZET2015/peak_Express_v1.json",
        'njobs' => 20,
        'globaltag' => $promptglobaltag,
        'apvmode' => 'peak',
        'submit' => $LA
        },
        
    'Cosmics0T_cruzet15_PromptReco_full_stats_part2' => {
        'py' => "$ENV{'PWD'}/alignment_cosmics_0T.py",
        'data' => "${datasetdir}/Cosmics2015/CRUZETandCRAFT2015/Cosmics_236370-238092_CRUZET_Commissioning2015-TkAlCosmics0T-PromptReco.txt",
        'json' => "${jsondir}/Cosmics2015/CRUZET2015/peak_Express_v1.json",
        'njobs' => 20,
        'globaltag' => $promptglobaltag,
        'apvmode' => 'peak',
        'submit' => $LA
        },
    'Cosmics3dot8T_craft15' => {
        'py' => "$ENV{'PWD'}/alignment_cosmics_forcefield38T.py",
        'data' => "${datasetdir}/Cosmics2015/CRUZETandCRAFT2015/Cosmics_238443-239517_CRAFT_Commissioning2015-TkAlCosmics0T-PromptReco.txt",
        'json' => "${jsondir}/Cosmics2015/CRUZET2015/peak_Express_v1.json",
        'njobs' => 20,
        'globaltag' => $promptglobaltag,
        'apvmode' => 'peak',
        'submit' => $LA
        },

        );

#system "eos rm -r $mssDir";
system "/afs/cern.ch/project/eos/installation/0.3.84-aquamarine/bin/eos.select mkdir $mssDir";

my $first = 1;

my $nconfs = keys(%confhash);

my $i = 0;

foreach my $key (keys %confhash) {

my $submit = $confhash{$key}->{'submit'};
my $json =  $confhash{$key}->{'json'};
my $globaltag = $confhash{$key}->{'globaltag'};
my $apvmode = $confhash{$key}->{'apvmode'};
my $primarywidth = $confhash{$key}->{'primarywidth'};
my $name = $key;
if ($submit) {

    print "start setting up $name\n";

    my $a = ' -a';
    $a = ''  if($first && !$append);
    $first = 0;

#mps_setup.pl [-m] [-a] milleScript cfgTemplate infiList nJobs class jobname [pedeScript [mssDir]]

my $cfgTemplate = $confhash{$key}->{'py'};
my $infiList = $confhash{$key}->{'data'};
my $nJobs = $confhash{$key}->{'njobs'};

print "\n\n";

unlink "tmp.py";

system "cp $cfgTemplate tmp.py";
$cfgTemplate = "tmp.py";


if(defined $json && $json ne "")
{


    unless (-e "$json")
    {
        print "json files was not found!\n";
        exit(-1);
    }
#my $jsoncontent = `cat $json`;
#my $st = 'perl -pi -e \'~s|#jsonfileplaceholder|' . $jsoncontent . '|\' tmp.py';
#system "$st\n";
&rplaceholder("tmp.py","$json","jsonfileplaceholder");
}
&rplaceholder("tmp.py","startgeometry.txt","placeholderstartgeometry");

#   my $startgeo = `cat startgeometry.txt`;
#   my $st = 'perl -pi -e \'~s|#placeholderstartgeometry|' . $startgeo . '|\' tmp.py';
#   system "$st\n";

&rplaceholder("tmp.py","deadmodules.txt","placeholderdeadmodules");

&rplaceholder("tmp.py","pedesettings.txt","placeholderpedesettings");

#&rplaceholder("tmp.py","determineLA.txt","placeholderdetermineLA") if ($LA);

&rplaceholder("tmp.py","alignables.txt","placeholderalignables");
# my $alignables = `cat alignables.txt`;
# $st = 'perl -pi -e \'~s|#placeholderalignables|' . $alignables . '|\' tmp.py';
# system "$st\n";

if(defined $globaltag && $globaltag ne "")
{
    my $st = 'perl -pi -e \'~s|process.GlobalTag.globaltag = \".+\"|process.GlobalTag.globaltag = \"' . $globaltag . '\"|\' tmp.py';
    system "$st\n";
}

if(defined $apvmode && $apvmode ne "")
{#StoNcommands = cms.vstring("ALL 12.0")
my $mode = "12.0";
$mode = "18.0" if($apvmode eq "peak");


my $st = 'perl -pi -e \'~s|StoNcommands = cms\.vstring\(\"ALL .+\"\)|StoNcommands = cms\.vstring\(\"ALL ' . $mode . '\"\)|\' tmp.py';
system "$st\n";

if($apvmode eq "peak")
{
    print "setting peak mode\n";
#    &rplaceholder("tmp.py","peakLA.txt","placeholderLA");
}
}

my @tmpfile = `cat tmp.py`;

unlink "tmp.py";
open OUT, "> tmp.py";
print OUT "#datasetconfigname $name\n";



for(my $qwe = 0; $qwe <= $#tmpfile; $qwe++)
{
    chomp $tmpfile[$qwe];


    if($tmpfile[$qwe] =~ /ParticleProperties\.PrimaryWidth/)
    {
        if(defined $primarywidth && $primarywidth >0.0)
        {
            print OUT "process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryWidth = $primarywidth\n";
        }
        else
        {
            print OUT "$tmpfile[$qwe]\n";
        }
    }
    else
    {
        print OUT "$tmpfile[$qwe]\n";
    }


}

close OUT;

&startmps("perl ${mpsdir}/mps_setup.pl -m${a} -M $mem -N $name $millescript $cfgTemplate $infiList $nJobs $class $jobname $pedeScript cmscafuser:$mssDir");

unlink "tmp.py";
$i++;

print "stop setting up $name\n";

}
}



# system "sh wtx_cosmics.sh 28";
# system "sh wtx_ztomumu.sh 10";
#####################





sub startmps()
{
    my $st = shift;
    print "$st\n";
    system "$st";
}

sub rplaceholder()
{
    my $filename = shift;
    my $source = shift;
    my $v = shift;


    unless(-e "$source")
    {
        print "unable to find file $source\n";
        exit(-1);
    }

    my @s = `cat $source`;
    my @f = `cat $filename`;

    unlink "$filename";

    open OUT, ">$filename";
    for(my $i =0; $i<=$#f;$i++)
    {
        my $tmp = $f[$i];
        chomp $tmp;
        if($tmp =~ /\#$v/)
        {
            for(my $j =0; $j<=$#s;$j++)
            {
                my $tmp_s = $s[$j];
                chomp $tmp_s;
                print OUT "$tmp_s\n";
            }
        }
        else
        {
            print OUT "$tmp\n";
        }
    }
    close OUT;
}
