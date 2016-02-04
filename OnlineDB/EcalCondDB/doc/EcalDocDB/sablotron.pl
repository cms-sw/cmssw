	#!/usr/bin/perl
	
	#FIXME: very important: how does NEW work in perl????????????????

	#use strict;
	use XML::Sablotron;
	use XML::XPath;
	use Text::Trim;
	use XML::Writer;

	my $processor = new XML::Sablotron();

	$processor->runProcessor("doc.xslt",
				"doc.xml",
				'arg:/result',
				undef, undef);

	open (HTML, ">", "EcalDocDB.html") or die $!;

	$XML = $processor->getResultArg("arg:/result");

	print HTML $XML;
	close HTML;
