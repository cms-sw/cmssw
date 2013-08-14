%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Name:     cmspaper.cls                                         %
%                                                                 %
%  Purpose:  Set LaTeX parameters for CMS note.                   %
%                                                                 %
%  This files is based on the Document Style 'CERNART'            %
%  by  Michel Goossens CN/ASD                                     %
%                                                                 %
%   - requires packages: graphics                                 %
%   - special treatment of title page                             %
%                                                                 %
%  Last mod : 25.11.1996                                          %
%  Updated for new CMS logo May 97 / JPP                          %
%  Updated for smaller page size Vov 00 / Lucas                   %
%                                                                 %
%  Grzegorz Wrochna, CERN/PPE                                     %
%                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\def\fileversion{0.4}\def\filedate{25 Nov. 1996}
\typeout{CMS paper style Version <\fileversion>, <\filedate>}
\typeout{based on Document Style 'CERNART' Version <2.1>, <1 May 1995>}
% -------------------------------- identification ---------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cmspaper}[1996/11/25]
% ---------------------------  declaration of options --
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
% ----------------------------  execution of options ------------------------
\ProcessOptions
% --------------------------------package loading ---------------------------
\LoadClass{article}
 
%%%%%%%% page size to be printable on both A4 and US Letter %%%%%%%

\setlength{\hoffset}{-1in}
%Lucas-setlength{\voffset}{-0.5in}
      \setlength{\voffset}{-1in}
\setlength{\oddsidemargin}{25mm}
\setlength{\evensidemargin}{25mm}
\setlength{\marginparwidth}{25mm}
\setlength{\marginparsep}{\baselineskip}
\setlength{\topmargin}{20mm}
\setlength{\headheight}{0mm}
\setlength{\headsep}{0mm}
%Lucas-setlength{\footskip}{6mm}
      \setlength{\footskip}{9mm}
\setlength{\textwidth}{16cm}
\ifcase \@ptsize
%JPP-\setlength{\textheight}{58\baselineskip}
     \setlength{\textheight}{57\baselineskip}
\or
%JPP-\setlength{\textheight}{52\baselineskip}
     \setlength{\textheight}{51\baselineskip}
\or 
%JPP-\setlength{\textheight}{47\baselineskip}
     \setlength{\textheight}{46\baselineskip}
\fi
\addtolength{\textheight}{\topskip}

%%%%% floats handling parameters for better placing of figures %%%%%%%
\renewcommand{\topfraction}{1.0}
\renewcommand{\dbltopfraction}{1.0}
\renewcommand{\bottomfraction}{1.0}
\renewcommand{\textfraction}{0.2}
\setlength{\intextsep}{6pt}
\setlength{\floatsep}{6pt}
\setlength{\textfloatsep}{6pt}
\setlength{\dblfloatsep}{6pt}
\setlength{\dbltextfloatsep}{6pt}

%%%%%%%%%%%%%%%% nonindented paragraphs %%%%%%%%%%%%%%%
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5\baselineskip}
 
%%%%%%%%%%%%%%%%%%%%% section headings %%%%%%%%%%%%%%%%%%%%%%%%%%

\def\section{\@startsection {section}{1}{\z@}%
{ 1\baselineskip plus  .5\baselineskip minus  .25\baselineskip}%
{+0.1pt plus +0.1pt minus +0.1pt}{\reset@font\Large\bfseries}}
 
\def\subsection{\@startsection{subsection}{2}{\z@}%
{ 1\baselineskip plus  .5\baselineskip minus  .25\baselineskip}%
{+0.1pt plus +0.1pt minus +0.1pt}{\reset@font\large\bfseries}}
 
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}%
{ 1\baselineskip plus  .5\baselineskip minus  .25\baselineskip}%
{+0.1pt plus +0.1pt minus +0.1pt}{\reset@font\normalsize\bfseries}}

%%%%%%%%%%%%%%%% define PostScript fonts %%%%%%%%%%%%%%%
  \renewcommand{\sfdefault}{phv}        % sansserif  = helvetica
  \renewcommand{\rmdefault}{ptm}        % roman      = times
  \renewcommand{\ttdefault}{pcr}        % typewriter = courier
  \newcommand{\hm}{\sf}                 % helvetica medium 
  \newcommand{\hb}{\bfseries\sffamily}  % helvetica bold
  \newcommand{\Hb}{\hb\fontsize{44pt}{46pt}\selectfont}  % HUGE helvetica bold

%%%%%%%%%%%%%%%% two floating figures, side by side %%%%%%%%%%%%%%%
\newenvironment{2figures}[1]{\begin{figure}[#1] 
  \begin{center}
    \begin{tabular}{p{.47\textwidth}p{.47\textwidth}} }
 {  \end{tabular}
  \end{center} 
 \end{figure}
}

%%%%%%%%%%%%%%%%%%%%%% footnotes %%%%%%%%%%%%%%%%%%%%%
\def\footnoterule{\kern-3\p@
 \hrule width .4\columnwidth
 \kern 2.6\p@}
 
%%%%%%%%%%%% Environment makefntext %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\long\def\@makefntext#1{\parindent=\z@
\begin{list}{}%
   {\labelwidth1.5em\leftmargin\labelwidth% set global width
    \labelsep3pt%  horizontal separation between term and item
    \itemsep0pt%   vertical separation between two items
    \parsep0pt%    vertical separation two paragraphs in an item
    \topsep0pt%    vertical separation text/list
    \small}
%   \footnotesize}
    \item [\hfill $^{\mathrm{\@thefnmark)}}$] #1\end{list}}% ***** end of makefntext
 
%%%%%%%%%%%% Environment Figlist %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newenvironment{Figlist}[1]{% Give highest figure number as parameter
\vskip2mm\begin{center}{\bf FIGURE CAPTIONS}\end{center}
\begin{list}{\bf Fig.\hfill\arabic{figure} : }{\usecounter{figure}
    \settowidth{\labelwidth}{\bf Fig . #1 : }% define global width
    \leftmargin\labelwidth \labelsep0pt
    \itemsep0pt plus 0.1ex minus 0.1ex
    \parsep0pt plus 0.1ex minus 0.1ex
    \topsep\baselineskip}}%
{\end{list}\setcounter{figure}{0}% Reset figure counter
}% ***** end of environment{Figlist} ****

%%%%%%%%%%%% Environment CERNfig %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newenvironment{CERNfig}{\begingroup % figures, one per page
\def\@makecaption##1##2{\hbox to\hsize{\hfil##1\hfil}}}%
{\endgroup}
% ***** end of environment{CERNfig} ****
 
%%%%%%%%%%%% Environment Tablist %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newenvironment{Tablist}[1]{% Give highest table number as parameter
\vskip2mm\begin{center}{\bf TABLE CAPTIONS}\end{center}
\begin{list}{\bf Tab.\hfill\arabic{table} : }{\usecounter{table}
    \settowidth{\labelwidth}{\bf Fig . #1 : }% define global width
    \leftmargin\labelwidth \labelsep0pt
    \itemsep0pt plus 0.1ex minus 0.1ex
    \parsep0pt plus 0.1ex minus 0.1ex
    \topsep\baselineskip}}%
{\end{list}\setcounter{table}{0}% Reset table counter
}% ***** end of environment{Tablist} ****
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% \begin{titlepage}
%     \cmsnote{yyyy-xxx}
%     \date{}
%     \title{ first  line \\
%             second line \\
%             etc...}
%     \author{B.B.Auth\Instref{xx} M.M.~Auth2\Bothref{I1}{A2}
%             C.D.user\Instref{yy} ...}
% \submitted{}
% \collaboration{}
% \conference{}
% \note{}
% \dedication{}
%     \Instfoot{xx}{text text}
%     \Instfoot{yy}{text text}
% \end{titlepage}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% title page %%%%%%%%%%%%%%%%%%%%%%%%%5
\def\maketitle{% Dummy maketitle
\typeout{Dummy maketitle --- Use titlepage instead}}

% Definition of environment "titlepage"
\def\titlepage{\clearpage%
%JPP-or maybe pagestyle{plain}  ??
\setcounter{footnote}{0}\pagestyle{empty}}% End of \titlepage tag

% CMS NOTE header \cmsnote
\def\cmsnote#1{{Available on CMS information server
\hfill\Large\bf CMS NOTE #1} 
\begin{center}\includegraphics{cms_note.eps}\end{center}\vspace*{0mm}}

% CMS INTERNAL NOTE header \internalnote
\def\internalnote#1{{Available on CMS information server
\hfill\Large\bf CMS IN #1} 
\begin{center}\includegraphics{cms_in.eps}\end{center}}

% CMS CONFERENCE REPORT header \conferencereport
\def\conferencereport#1{{Available on CMS information server
\hfill\Large\bf CMS CR #1} 
\begin{center}\includegraphics{cms_cr.eps}\end{center}}

% Date (flushright)
\def\date#1{{\large\bf\hfill #1}}
%\def\date#1{\edef\@temp{#1}\ifx\@temp\@empty\def\@temp{\today}\fi
%\hbox to \hsize{\hskip123mm\hbox{\@temp}\hss}}
% Submitted{}
\def\submitted#1{\vskip1em\begin{center}#1\end{center}}
% collaboration{}
\def\collaboration#1{\vskip1em\begin{center}\bfseries\itshape#1\end{center}}
% conference{}
\def\conference#1{\vskip1em\begin{center}#1\end{center}}
% note{}
\def\note#1{\vskip1em\begin{center}#1\end{center}}
% dedication{}
\def\dedication#1{\vskip1em\begin{center}\it#1\end{center}}
% Title
\def\title#1{\vskip1.5cm\begin{center}\huge\sf#1\end{center}\vskip1.5em}
% Author
%\def\@makefnmark{\hbox to 0pt{$^{\@thefnmark)\,}$\hss}}
\def\@makefnmark{\hbox{$^{\@thefnmark)}$}}
\def\author#1{%% Treat the list of authors
\setcounter{footnote}{0}\def\@currentlabel{}%
\begingroup\def\thefootnote{\arabic{footnote}}
\def\@makefnmark{\hbox{$^{\rm\@thefnmark)}$}}
\global\@topnum\z@ \begin{center}{\lineskip.5em
\begin{tabular}[t]{c}#1\end{tabular}\par}
\end{center}\par\vskip1.5em\@thanks\endgroup}
%
\newenvironment{Authlist}{\center}{\endcenter}
% Abstract
\def\abstract{\begin{center}{\vskip1.5em\bf Abstract}\\[\baselineskip]%
\end{center}\quotation\noindent%
\setlength{\parindent}{0pt}%
\setlength{\parskip}{0.5\baselineskip}%
\ignorespaces}
\def\endabstract{\endquotation\par}
%
%%%%%%%%%%%% Institute definitions  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\@ifundefined{c@instfoot}{\newcounter{instfoot}}{}
\def\Iref#1{$^{\Inst{#1})}$}   \let\Instref=\Iref
\def\Inst#1{\@ifundefined{r@#1}{{\bf ??}}%
{\edef\@tempa{\@nameuse{r@#1}}\expandafter\@car\@tempa \@nil\null}}
%
%%AS If we found `Instfoot' inside `Authlist' env, we will changes the
%    formatting.
%
\def\Instfoot#1#2{\def\@temp{Authlist}%
\ifx\@temp\@currenvir
    \par {\small\it#2} \\[2ex]%
\else
    \refstepcounter{instfoot}\label{#1}%
    \xdef\@thefnmark{\arabic{instfoot}}\@footnotetext{#2}%
\fi
}
%
%%%%%%%%%%%% Anotation definitions  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%\@ifundefined{c@anotfoot}{\newcounter{anotfoot}}{}
\@ifundefined{mathrm}{\newcommand{\mathrm}[1]{{\rm#1}}}{}
\def\Aref#1{$^{\mathrm{#1})}$} \let\Anotref=\Aref
\def\Anotfoot#1#2{\xdef\@thefnmark{#1}\@footnotetext{#2}}
%
%%%%%%%%%%%% Anotation references   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\AAref#1#2{$^{\mathrm{#1,#2)}}$}       \let\AnotAnotref=\AAref
\def\IAref#1#2{$^{\Inst{#1},\mathrm{#2)}}$}\let\InstAnotref=\IAref
\def\IIref#1#2{$^{\Inst{#1},\Inst{#2})}$}  \let\InstInstref=\IIref
\def\IIIref#1#2#3{$^{\Inst{#1},\Inst{#2},\Inst{#3})}$}
\def\IIAref#1#2#3{$^{\Inst{#1},\Inst{#2},\mathrm{#3)}}$}
%%%%%%%%%%%% Annospec environment   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newenvironment{Annospec}[1]{% Give width for offset of text
\vskip2mm\bgroup
%\def\@makefnmark{\hbox{$^{\mathrm{\@thefnmark)}}$}}
\long\def\@makefntext##1{\parindent=\z@
\begin{list}{}%
   {\labelwidth=#1\leftmargin\labelwidth% set global width
    \labelsep3pt%  horizontal separation between term and item
    \itemsep0pt%   vertical separation between two items
    \parsep0pt%    vertical separation two paragraphs in an item
    \topsep0pt%    vertical separation text/list
    \small}
%   \footnotesize}
    \item [\hfill $^{\mathrm{\@thefnmark)}}$] ##1\end{list}}}% ***** end of makefntext
{\egroup}% ***** end of environment{Annospec} *****
 
 
% End of titlepage
\def\endtitlepage{%% Reset counters
\setcounter{footnote}{0}\let\titlepage\relax\vfill
%JPP-this is not the proper place for secounter{page}, must be in the tex file
%JPP-\newpage\setcounter{page}{1}\pagestyle{plain}\pagenumbering{arabic}%
\newpage\pagestyle{plain}\pagenumbering{arabic}%
\gdef\@thanks{}\gdef\@author{}\gdef\@title{}\let\thanks\relax}

\RequirePackage{graphics}
 
% Local CERN stuff
 
\newcommand{\bra}[1]{\left\langle #1\right|} % <bra|
\newcommand{\ket}[1]{\left| #1\right\rangle} % |ket>
\newcommand{\braket}[2]{\left\langle#1 | #2\right\rangle} % <bra|ket>
 
\endinput
 
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
