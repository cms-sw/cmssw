#ifndef Fireworks_Core_FWDialogBuilder_h
#define Fireworks_Core_FWDialogBuilder_h

#include "TGNumberEntry.h"

class TGCompositeFrame;
class TGLayoutHints;
class TGVerticalFrame;
class TGLabel;
class FWColorSelect;
class TGTextView;
class TGTextEntry;
class TGTextButton;
class TGHSlider;
class FWTableManagerBase;
class FWTableWidget;
class TGCheckButton;
class FWGUIValidatingTextEntry;
class TGTab;
class FWColorManager;
class TGTextEdit;
class TGHtml;

class FWLayoutBuilder
{
protected:
   FWLayoutBuilder(TGCompositeFrame *window, bool expandY = true);
   FWLayoutBuilder &newRow();

   FWLayoutBuilder &indent(int left = 2, int right = -1);
   
   FWLayoutBuilder &unindent(void);
   TGCompositeFrame *currentFrame(void) { return m_currentFrame; }
   FWLayoutBuilder &floatLeft(size_t spacing);
   FWLayoutBuilder &spaceUp(size_t spacing);
   FWLayoutBuilder &spaceDown(size_t spacing);
   FWLayoutBuilder &spaceLeft(size_t spacing);
   FWLayoutBuilder &spaceRight(size_t spacing);
   FWLayoutBuilder &frameSpaceUp(size_t spacing);
   FWLayoutBuilder &frameSpaceDown(size_t spacing);
   FWLayoutBuilder &frameSpaceLeft(size_t spacing);
   FWLayoutBuilder &frameSpaceRight(size_t spacing);
   FWLayoutBuilder &expand(bool expandX = true, bool expandY = false);

   bool isFloatingLeft() { return m_floatLeft; }
   TGLayoutHints *nextHints();
   TGCompositeFrame *nextFrame();
   TGVerticalFrame  *verticalFrame();
   void            frameForTab();

private:   
   TGCompositeFrame *m_window;

   std::vector<TGVerticalFrame *> m_framesStack;
   //  TGCompositeFrame *m_lastFrame;
   TGCompositeFrame *m_currentFrame;

   bool             m_floatLeft;
   size_t           m_topSpacing;
   size_t           m_leftSpacing;
   TGLayoutHints   *m_currentHints;
   TGLayoutHints   *m_currentFrameHints;
};

/** Helper class to construct dialogs in a more readable ways.

    Encapsulated TGUI layout hiccups and exposes the developer an API which
    allows to layout items in a top->bottom, right->left manner.
    
    Example:
    
      FWDialogBuilder builder(parent);
      parent.newRow(2)              // New row which has a 2 pixel padding on top.
            .addLabel("MyLabel:")    // A new label.
            .indent(20)             // Whatever follows is indented 20 pixels 
                                    // on the right.
            .addLabel("MyLabel2")   // Another label.
            .spaceDown(4)
            .addTextButton("Aligned to MyLabel2 ").floatLeft()
            .addTextButton("Same Row as previous")
            .unindent()              // back one level in the indentation.
            .addLabel("Aligned to MyLabel:")
            
    Because in ROOT layout and parenting of widgets are mixed we need to take
    responsibility for creating the widget objects (sigh!), so we have one
    "addXYZ" method per widget that can be added. If we find our way around
    this it would be better to have a generic "addWidget()" method and create
    widgets outside this class.
    
    TODO: For higher configurability we should have an 
          "addWithCallback(Callbak)"  method which can be used to specify a 
          generic widget creation action.
  */
class FWDialogBuilder : public FWLayoutBuilder
{
public:
   FWDialogBuilder(TGCompositeFrame *window, FWDialogBuilder *parent = 0, bool expandY = true);

   FWDialogBuilder &newRow();
   FWDialogBuilder &indent(int left = 2, int right = -1);   
   FWDialogBuilder &unindent(void);
   
   FWDialogBuilder &addLabel(const char *text,
                             size_t fontSize = 12,
                             size_t weight = 0,
                             TGLabel **out = 0);
   
   FWDialogBuilder &addTextView(const char *defaultText = 0,
                                TGTextView **out = 0);

  // Is default text meaningful here as the html is
  // a document with structure?
   FWDialogBuilder &addHtml(TGHtml **out = 0);

   FWDialogBuilder &addTextEdit(const char *defaultText = 0,
                                TGTextEdit **out = 0);
   FWDialogBuilder &addColorPicker(const FWColorManager *manager,
                                   FWColorSelect **out = 0);
   
   FWDialogBuilder &addHSlider(size_t size, TGHSlider **out = 0);
   
   FWDialogBuilder &addTextButton(const char *text, TGTextButton **out = 0);
   FWDialogBuilder &addValidatingTextEntry(const char *defaultText, 
                                           FWGUIValidatingTextEntry **out);
   FWDialogBuilder &addTextEntry(const char *defaultText, TGTextEntry **out);
   FWDialogBuilder &addNumberEntry(float defaultValue, size_t digits,
                                   TGNumberFormat::EStyle style,
                                   int min, int max,
                                   TGNumberEntry **out);
   
   FWDialogBuilder &addCheckbox(const char *text, TGCheckButton **out = 0);
   FWDialogBuilder &addTable(FWTableManagerBase *manager, FWTableWidget **out = 0);

      
   FWDialogBuilder &addHSeparator(size_t horizontalPadding = 4, 
                                  size_t verticalPadding = 3);

   FWDialogBuilder &tabs(TGTab **out);
   FWDialogBuilder &untabs(void);
   FWDialogBuilder &beginTab(const char *label);
   FWDialogBuilder &endTab(void);
   
   FWDialogBuilder &floatLeft(size_t spacing = 3);

   FWDialogBuilder &spaceUp(size_t spacing = 3);
   FWDialogBuilder &spaceDown(size_t spacing = 3);
   FWDialogBuilder &spaceUpDown(size_t spacing = 3);
   FWDialogBuilder &spaceLeft(size_t spacing = 3);
   FWDialogBuilder &spaceRight(size_t spacing = 3);
   FWDialogBuilder &spaceLeftRight(size_t spacing = 3);

   FWDialogBuilder &frameSpaceUp(size_t spacing = 3);
   FWDialogBuilder &frameSpaceDown(size_t spacing = 3);
   FWDialogBuilder &frameSpaceUpDown(size_t spacing = 3);
   FWDialogBuilder &frameSpaceLeft(size_t spacing = 3);
   FWDialogBuilder &frameSpaceRight(size_t spacing = 3);
   FWDialogBuilder &frameSpaceLeftRight(size_t spacing = 3);

   FWDialogBuilder &expand(size_t expandX = true, size_t expandY = false);
   FWDialogBuilder &vSpacer(size_t size = 0);
   FWDialogBuilder &hSpacer(size_t size = 0);

protected:
   template <class T> FWDialogBuilder &extract(T *in, T **out)
      {
         if (out)
            *out = in;
         return *this;
      }
   
private:
   FWDialogBuilder *m_parent;
   TGTab           *m_tabs;
};

#endif
